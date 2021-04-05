// Copyright 2017 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Traits and implementations for writing bits to a stream.
//!
//! ## Example
//!
//! Writing the initial STREAMINFO block to a FLAC file,
//! as documented in its
//! [specification](https://xiph.org/flac/format.html#stream).
//!
//! ```
//! use tokio::io::{AsyncWrite, AsyncWriteExt};
//! use bitstream_io::{BigEndian, AsyncBitWriter, AsyncBitWrite};
//!
//! # tokio_test::block_on(async {
//! let mut flac: Vec<u8> = Vec::new();
//! {
//!     let mut writer = AsyncBitWriter::endian(&mut flac, BigEndian);
//!
//!     // stream marker
//!     writer.write_bytes(b"fLaC").await.unwrap();
//!
//!     // metadata block header
//!     let last_block: bool = false;
//!     let block_type: u8 = 0;
//!     let block_size: u32 = 34;
//!     writer.write_bit(last_block).await.unwrap();
//!     writer.write(7, block_type).await.unwrap();
//!     writer.write(24, block_size).await.unwrap();
//!
//!     // STREAMINFO block
//!     let minimum_block_size: u16 = 4096;
//!     let maximum_block_size: u16 = 4096;
//!     let minimum_frame_size: u32 = 1542;
//!     let maximum_frame_size: u32 = 8546;
//!     let sample_rate: u32 = 44100;
//!     let channels: u8 = 2;
//!     let bits_per_sample: u8 = 16;
//!     let total_samples: u64 = 304844;
//!     writer.write(16, minimum_block_size).await.unwrap();
//!     writer.write(16, maximum_block_size).await.unwrap();
//!     writer.write(24, minimum_frame_size).await.unwrap();
//!     writer.write(24, maximum_frame_size).await.unwrap();
//!     writer.write(20, sample_rate).await.unwrap();
//!     writer.write(3, channels - 1).await.unwrap();
//!     writer.write(5, bits_per_sample - 1).await.unwrap();
//!     writer.write(36, total_samples).await.unwrap();
//! }
//!
//! // STREAMINFO's MD5 sum
//!
//! // Note that the wrapped writer can be used once bitstream writing
//! // is finished at exactly the position one would expect.
//!
//! flac.write_all(
//!     b"\xFA\xF2\x69\x2F\xFD\xEC\x2D\x5B\x30\x01\x76\xB4\x62\x88\x7D\x92")
//!     .await
//!     .unwrap();
//!
//! assert_eq!(flac, vec![0x66,0x4C,0x61,0x43,0x00,0x00,0x00,0x22,
//!                       0x10,0x00,0x10,0x00,0x00,0x06,0x06,0x00,
//!                       0x21,0x62,0x0A,0xC4,0x42,0xF0,0x00,0x04,
//!                       0xA6,0xCC,0xFA,0xF2,0x69,0x2F,0xFD,0xEC,
//!                       0x2D,0x5B,0x30,0x01,0x76,0xB4,0x62,0x88,
//!                       0x7D,0x92]);
//! # });
//! ```

#![warn(missing_docs)]

use std::convert::From;
use std::io;
use std::ops::{AddAssign, Rem};
use async_trait::async_trait;

use tokio::io::{AsyncWrite, AsyncWriteExt};

use super::{huffman::WriteHuffmanTree, BitQueue, AsyncEndianness, Numeric, PhantomData, SignedNumeric};

/// For writing bit values to an underlying stream in a given endianness.
///
/// Because this only writes whole bytes to the underlying stream,
/// it is important that output is byte-aligned before the bitstream
/// writer's lifetime ends.
/// **Partial bytes will be lost** if the writer is disposed of
/// before they can be written.
pub struct AsyncBitWriter<W: AsyncWrite + Unpin + Sync + Send, E: AsyncEndianness + Sync + Send> {
    writer: W,
    bitqueue: BitQueue<E, u8>,
}

impl<W: AsyncWrite + Unpin + Sync + Send, E: AsyncEndianness + Sync + Send> AsyncBitWriter<W, E> {
    /// Wraps an AsyncBitWriter around something that implements `AsyncWrite`
    pub fn new(writer: W) -> AsyncBitWriter<W, E> {
        AsyncBitWriter {
            writer,
            bitqueue: BitQueue::new(),
        }
    }

    /// Wraps an AsyncBitWriter around something that implements `AsyncWrite`
    /// with the given endianness.
    pub fn endian(writer: W, _endian: E) -> AsyncBitWriter<W, E> {
        AsyncBitWriter {
            writer,
            bitqueue: BitQueue::new(),
        }
    }

    /// Unwraps internal writer and disposes of AsyncBitWriter.
    ///
    /// # Warning
    ///
    /// Any unwritten partial bits are discarded.
    #[inline]
    pub fn into_writer(self) -> W {
        self.writer
    }

    /// If stream is byte-aligned, provides mutable reference
    /// to internal writer.  Otherwise returns `None`
    #[inline]
    pub fn writer(&mut self) -> Option<&mut W> {
        if self.byte_aligned() {
            Some(&mut self.writer)
        } else {
            None
        }
    }

    /// Converts `BitWriter` to `ByteWriter` in the same endianness.
    ///
    /// # Warning
    ///
    /// Any written partial bits are discarded.
    #[inline]
    pub fn into_bytewriter(self) -> AsyncByteWriter<W, E> {
        AsyncByteWriter::new(self.into_writer())
    }

    /// If stream is byte-aligned, provides temporary `ByteWriter`
    /// in the same endianness.  Otherwise returns `None`
    ///
    /// # Warning
    ///
    /// Any unwritten bits left over when `ByteWriter` is dropped are lost.
    #[inline]
    pub fn bytewriter(&mut self) -> Option<AsyncByteWriter<&mut W, E>> {
        self.writer().map(AsyncByteWriter::new)
    }

    /// Consumes writer and returns any un-written partial byte
    /// as a `(bits, value)` tuple.
    ///
    /// # Examples
    /// ```
    /// use tokio::io::AsyncWrite;
    /// use bitstream_io::{BigEndian, AsyncBitWriter, AsyncBitWrite};
    /// # tokio_test::block_on(async {
    /// let mut data = Vec::new();
    /// let (bits, value) = {
    ///     let mut writer = AsyncBitWriter::endian(&mut data, BigEndian);
    ///     writer.write(15, 0b1010_0101_0101_101).await.unwrap();
    ///     writer.into_unwritten()
    /// };
    /// assert_eq!(data, [0b1010_0101]);
    /// assert_eq!(bits, 7);
    /// assert_eq!(value, 0b0101_101);
    /// # });
    /// ```
    ///
    /// ```
    /// use tokio::io::AsyncWrite;
    /// use bitstream_io::{BigEndian, AsyncBitWriter, AsyncBitWrite};
    /// # tokio_test::block_on(async {
    /// let mut data = Vec::new();
    /// let (bits, value) = {
    ///     let mut writer = AsyncBitWriter::endian(&mut data, BigEndian);
    ///     writer.write(8, 0b1010_0101).await.unwrap();
    ///     writer.into_unwritten()
    /// };
    /// assert_eq!(data, [0b1010_0101]);
    /// assert_eq!(bits, 0);
    /// assert_eq!(value, 0);
    /// # });
    /// ```
    #[inline(always)]
    pub fn into_unwritten(self) -> (u32, u8) {
        (self.bitqueue.len(), self.bitqueue.value())
    }

    /// Flushes output stream to disk, if necessary.
    /// Any partial bytes are not flushed.
    ///
    /// # Errors
    ///
    /// Passes along any errors from the underlying stream.
    #[inline(always)]
    pub async fn flush(&mut self) -> io::Result<()> {
        self.writer.flush().await
    }
}

/// A trait for anything that can write a variable number of
/// potentially un-aligned values to an output stream
#[async_trait]
pub trait AsyncBitWrite: Sync + Send {
    /// Writes a single bit to the stream.
    /// `true` indicates 1, `false` indicates 0
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underlying stream.
    async fn write_bit(&mut self, bit: bool) -> io::Result<()>;

    /// Writes an unsigned value to the stream using the given
    /// number of bits.
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underlying stream.
    /// Returns an error if the input type is too small
    /// to hold the given number of bits.
    /// Returns an error if the value is too large
    /// to fit the given number of bits.
    async fn write<U>(&mut self, bits: u32, value: U) -> io::Result<()>
    where
        U: Numeric + Sync + Send;

    /// Writes a twos-complement signed value to the stream
    /// with the given number of bits.
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underlying stream.
    /// Returns an error if the input type is too small
    /// to hold the given number of bits.
    /// Returns an error if the value is too large
    /// to fit the given number of bits.
    async fn write_signed<S>(&mut self, bits: u32, value: S) -> io::Result<()>
    where
        S: SignedNumeric + Sync + Send;

    /// Writes the entirety of a byte buffer to the stream.
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underlying stream.
    ///
    /// # Example
    ///
    /// ```
    /// use tokio::io::AsyncWrite;
    /// use bitstream_io::{BigEndian, AsyncBitWriter, AsyncBitWrite};
    /// # tokio_test::block_on(async {
    /// let mut writer = AsyncBitWriter::endian(Vec::new(), BigEndian);
    /// writer.write(8, 0x66).await.unwrap();
    /// writer.write(8, 0x6F).await.unwrap();
    /// writer.write(8, 0x6F).await.unwrap();
    /// writer.write_bytes(b"bar").await.unwrap();
    /// assert_eq!(writer.into_writer(), b"foobar");
    /// # });
    /// ```
    #[inline]
    async fn write_bytes(&mut self, buf: &[u8]) -> io::Result<()> {
        for b in buf.iter() {
            self.write(8, *b).await?;
        }

        Ok(())
    }

    /// Writes `value` number of 1 bits to the stream
    /// and then writes a 0 bit.  This field is variably-sized.
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underyling stream.
    ///
    /// # Examples
    /// ```
    /// use tokio::io::AsyncWrite;
    /// use bitstream_io::{BigEndian, AsyncBitWriter, AsyncBitWrite};
    /// # tokio_test::block_on(async {
    /// let mut writer = AsyncBitWriter::endian(Vec::new(), BigEndian);
    /// writer.write_unary0(0).await.unwrap();
    /// writer.write_unary0(3).await.unwrap();
    /// writer.write_unary0(10).await.unwrap();
    /// assert_eq!(writer.into_writer(), [0b01110111, 0b11111110]);
    /// # });
    /// ```
    ///
    /// ```
    /// use tokio::io::AsyncWrite;
    /// use bitstream_io::{LittleEndian, AsyncBitWriter, AsyncBitWrite};
    /// # tokio_test::block_on(async {
    /// let mut writer = AsyncBitWriter::endian(Vec::new(), LittleEndian);
    /// writer.write_unary0(0).await.unwrap();
    /// writer.write_unary0(3).await.unwrap();
    /// writer.write_unary0(10).await.unwrap();
    /// assert_eq!(writer.into_writer(), [0b11101110, 0b01111111]);
    /// # });
    /// ```
    async fn write_unary0(&mut self, value: u32) -> io::Result<()> {
        match value {
            0 => self.write_bit(false).await,
            bits @ 1..=31 => {
                self.write(value, (1u32 << bits) - 1).await?;
                self.write_bit(false).await
            }
            32 => {
                self.write(value, 0xFFFF_FFFFu32).await?;
                self.write_bit(false).await
            }
            bits @ 33..=63 => {
                self.write(value, (1u64 << bits) - 1).await?;
                self.write_bit(false).await
            }
            64 => {
                self.write(value, 0xFFFF_FFFF_FFFF_FFFFu64).await?;
                self.write_bit(false).await
            }
            mut bits => {
                while bits > 64 {
                    self.write(64, 0xFFFF_FFFF_FFFF_FFFFu64).await?;
                    bits -= 64;
                }
                self.write_unary0(bits).await
            }
        }
    }

    /// Writes `value` number of 0 bits to the stream
    /// and then writes a 1 bit.  This field is variably-sized.
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underyling stream.
    ///
    /// # Example
    /// ```
    /// use tokio::io::AsyncWrite;
    /// use bitstream_io::{BigEndian, AsyncBitWriter, AsyncBitWrite};
    /// # tokio_test::block_on(async {
    /// let mut writer = AsyncBitWriter::endian(Vec::new(), BigEndian);
    /// writer.write_unary1(0).await.unwrap();
    /// writer.write_unary1(3).await.unwrap();
    /// writer.write_unary1(10).await.unwrap();
    /// assert_eq!(writer.into_writer(), [0b10001000, 0b00000001]);
    /// # });
    /// ```
    ///
    /// ```
    /// use tokio::io::AsyncWrite;
    /// use bitstream_io::{LittleEndian, AsyncBitWriter, AsyncBitWrite};
    /// # tokio_test::block_on(async {
    /// let mut writer = AsyncBitWriter::endian(Vec::new(), LittleEndian);
    /// writer.write_unary1(0).await.unwrap();
    /// writer.write_unary1(3).await.unwrap();
    /// writer.write_unary1(10).await.unwrap();
    /// assert_eq!(writer.into_writer(), [0b00010001, 0b10000000]);
    /// # });
    /// ```
    async fn write_unary1(&mut self, value: u32) -> io::Result<()> {
        match value {
            0 => self.write_bit(true).await,
            1..=32 => {
                self.write(value, 0u32).await?;
                self.write_bit(true).await
            }
            33..=64 => {
                self.write(value, 0u64).await?;
                self.write_bit(true).await
            }
            mut bits => {
                while bits > 64 {
                    self.write(64, 0u64).await?;
                    bits -= 64;
                }
                self.write_unary1(bits).await
            }
        }
    }

    /// Returns true if the stream is aligned at a whole byte.
    fn byte_aligned(&self) -> bool;

    /// Pads the stream with 0 bits until it is aligned at a whole byte.
    /// Does nothing if the stream is already aligned.
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underyling stream.
    ///
    /// # Example
    /// ```
    /// use tokio::io::AsyncWrite;
    /// use bitstream_io::{BigEndian, AsyncBitWriter, AsyncBitWrite};
    /// # tokio_test::block_on(async {
    /// let mut writer = AsyncBitWriter::endian(Vec::new(), BigEndian);
    /// writer.write(1, 0).await.unwrap();
    /// writer.byte_align().await.unwrap();
    /// writer.write(8, 0xFF).await.unwrap();
    /// assert_eq!(writer.into_writer(), [0x00, 0xFF]);
    /// # });
    /// ```
    async fn byte_align(&mut self) -> io::Result<()> {
        while !self.byte_aligned() {
            self.write_bit(false).await?;
        }
        Ok(())
    }
}

/// A trait for anything that can write Huffman codes
/// of a given endianness to an output stream
#[async_trait]
pub trait AsyncHuffmanWrite<E: AsyncEndianness> {
    /// Writes Huffman code for the given symbol to the stream.
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underlying stream.
    async fn write_huffman<T>(&mut self, tree: &WriteHuffmanTree<E, T>, symbol: T) -> io::Result<()>
    where
        T: Ord + Copy + Sync + Send;
}

#[async_trait]
impl<W: AsyncWrite + Unpin + Sync + Send, E: AsyncEndianness + Sync + Send> AsyncBitWrite for AsyncBitWriter<W, E> {
    /// # Examples
    /// ```
    /// use tokio::io::AsyncWrite;
    /// use bitstream_io::{BigEndian, AsyncBitWriter, AsyncBitWrite};
    /// # tokio_test::block_on(async {
    /// let mut writer = AsyncBitWriter::endian(Vec::new(), BigEndian);
    /// writer.write_bit(true).await.unwrap();
    /// writer.write_bit(false).await.unwrap();
    /// writer.write_bit(true).await.unwrap();
    /// writer.write_bit(true).await.unwrap();
    /// writer.write_bit(false).await.unwrap();
    /// writer.write_bit(true).await.unwrap();
    /// writer.write_bit(true).await.unwrap();
    /// writer.write_bit(true).await.unwrap();
    /// assert_eq!(writer.into_writer(), [0b10110111]);
    /// # });
    /// ```
    ///
    /// ```
    /// use tokio::io::AsyncWrite;
    /// use bitstream_io::{LittleEndian, AsyncBitWriter, AsyncBitWrite};
    /// # tokio_test::block_on(async {
    /// let mut writer = AsyncBitWriter::endian(Vec::new(), LittleEndian);
    /// writer.write_bit(true).await.unwrap();
    /// writer.write_bit(true).await.unwrap();
    /// writer.write_bit(true).await.unwrap();
    /// writer.write_bit(false).await.unwrap();
    /// writer.write_bit(true).await.unwrap();
    /// writer.write_bit(true).await.unwrap();
    /// writer.write_bit(false).await.unwrap();
    /// writer.write_bit(true).await.unwrap();
    /// assert_eq!(writer.into_writer(), [0b10110111]);
    /// # });
    /// ```
    async fn write_bit(&mut self, bit: bool) -> io::Result<()> {
        self.bitqueue.push(1, if bit { 1 } else { 0 });
        if self.bitqueue.is_full() {
            write_byte(&mut self.writer, self.bitqueue.pop(8)).await
        } else {
            Ok(())
        }
    }

    /// # Examples
    /// ```
    /// use tokio::io::AsyncWrite;
    /// use bitstream_io::{BigEndian, AsyncBitWriter, AsyncBitWrite};
    /// # tokio_test::block_on(async {
    /// let mut writer = AsyncBitWriter::endian(Vec::new(), BigEndian);
    /// writer.write(1, 0b1).await.unwrap();
    /// writer.write(2, 0b01).await.unwrap();
    /// writer.write(5, 0b10111).await.unwrap();
    /// assert_eq!(writer.into_writer(), [0b10110111]);
    /// # });
    /// ```
    ///
    /// ```
    /// use tokio::io::AsyncWrite;
    /// use bitstream_io::{LittleEndian, AsyncBitWriter, AsyncBitWrite};
    /// # tokio_test::block_on(async {
    /// let mut writer = AsyncBitWriter::endian(Vec::new(), LittleEndian);
    /// writer.write(1, 0b1).await.unwrap();
    /// writer.write(2, 0b11).await.unwrap();
    /// writer.write(5, 0b10110).await.unwrap();
    /// assert_eq!(writer.into_writer(), [0b10110111]);
    /// # });
    /// ```
    ///
    /// ```
    /// use tokio::io::{AsyncWrite, sink};
    /// use bitstream_io::{BigEndian, AsyncBitWriter, AsyncBitWrite};
    /// # tokio_test::block_on(async {
    /// let mut w = AsyncBitWriter::endian(sink(), BigEndian);
    /// assert!(w.write(9, 0u8).await.is_err());    // can't write  u8 in 9 bits
    /// assert!(w.write(17, 0u16).await.is_err());  // can't write u16 in 17 bits
    /// assert!(w.write(33, 0u32).await.is_err());  // can't write u32 in 33 bits
    /// assert!(w.write(65, 0u64).await.is_err());  // can't write u64 in 65 bits
    /// assert!(w.write(1, 2).await.is_err());      // can't write   2 in 1 bit
    /// assert!(w.write(2, 4).await.is_err());      // can't write   4 in 2 bits
    /// assert!(w.write(3, 8).await.is_err());      // can't write   8 in 3 bits
    /// assert!(w.write(4, 16).await.is_err());     // can't write  16 in 4 bits
    /// # });
    /// ```
    async fn write<U>(&mut self, bits: u32, value: U) -> io::Result<()>
    where
        U: Numeric + Sync + Send,
    {
        if bits > U::bits_size() {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "excessive bits for type written",
            ))
        } else if (bits < U::bits_size()) && (value >= (U::one() << bits)) {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "excessive value for bits written",
            ))
        } else if bits < self.bitqueue.remaining_len() {
            self.bitqueue.push(bits, value.to_u8());
            Ok(())
        } else {
            let mut acc = BitQueue::from_value(value, bits);
            write_unaligned(&mut self.writer, &mut acc, &mut self.bitqueue).await?;
            write_aligned(&mut self.writer, &mut acc).await?;
            self.bitqueue.push(acc.len(), acc.value().to_u8());
            Ok(())
        }
    }

    /// # Examples
    /// ```
    /// use tokio::io::AsyncWrite;
    /// use bitstream_io::{BigEndian, AsyncBitWriter, AsyncBitWrite};
    /// # tokio_test::block_on(async {
    /// let mut writer = AsyncBitWriter::endian(Vec::new(), BigEndian);
    /// writer.write_signed(4, -5).await.unwrap();
    /// writer.write_signed(4, 7).await.unwrap();
    /// assert_eq!(writer.into_writer(), [0b10110111]);
    /// # });
    /// ```
    ///
    /// ```
    /// use tokio::io::AsyncWrite;
    /// use bitstream_io::{LittleEndian, AsyncBitWriter, AsyncBitWrite};
    /// # tokio_test::block_on(async {
    /// let mut writer = AsyncBitWriter::endian(Vec::new(), LittleEndian);
    /// writer.write_signed(4, 7).await.unwrap();
    /// writer.write_signed(4, -5).await.unwrap();
    /// assert_eq!(writer.into_writer(), [0b10110111]);
    /// # });
    /// ```
    #[inline]
    async fn write_signed<S>(&mut self, bits: u32, value: S) -> io::Result<()>
    where
        S: SignedNumeric + Sync + Send,
    {
        E::write_signed(self, bits, value).await
    }

    #[inline]
    async fn write_bytes(&mut self, buf: &[u8]) -> io::Result<()> {
        if self.byte_aligned() {
            self.writer.write_all(buf).await
        } else {
            for b in buf.iter() {
                self.write(8, *b).await?
            }

            Ok(())
        }
    }

    /// # Example
    /// ```
    /// use tokio::io::{AsyncWrite, sink};
    /// use bitstream_io::{BigEndian, AsyncBitWriter, AsyncBitWrite};
    /// # tokio_test::block_on(async {
    /// let mut writer = AsyncBitWriter::endian(sink(), BigEndian);
    /// assert_eq!(writer.byte_aligned(), true);
    /// writer.write(1, 0).await.unwrap();
    /// assert_eq!(writer.byte_aligned(), false);
    /// writer.write(7, 0).await.unwrap();
    /// assert_eq!(writer.byte_aligned(), true);
    /// # });
    /// ```
    #[inline(always)]
    fn byte_aligned(&self) -> bool {
        self.bitqueue.is_empty()
    }
}

#[async_trait]
impl<W: AsyncWrite + Unpin + Sync + Send, E: AsyncEndianness + Sync + Send> AsyncHuffmanWrite<E> for AsyncBitWriter<W, E> {
    /// # Example
    /// ```
    /// use tokio::io::AsyncWrite;
    /// use bitstream_io::{BigEndian, AsyncBitWriter, AsyncHuffmanWrite};
    /// use bitstream_io::huffman::compile_write_tree;
    /// let tree = compile_write_tree(
    ///     vec![('a', vec![0]),
    ///          ('b', vec![1, 0]),
    ///          ('c', vec![1, 1, 0]),
    ///          ('d', vec![1, 1, 1])]).unwrap();
    /// # tokio_test::block_on(async {
    /// let mut writer = AsyncBitWriter::endian(Vec::new(), BigEndian);
    /// writer.write_huffman(&tree, 'b').await.unwrap();
    /// writer.write_huffman(&tree, 'c').await.unwrap();
    /// writer.write_huffman(&tree, 'd').await.unwrap();
    /// assert_eq!(writer.into_writer(), [0b10110111]);
    /// # });
    /// ```
    #[inline]
    async fn write_huffman<T>(&mut self, tree: &WriteHuffmanTree<E, T>, symbol: T) -> io::Result<()>
    where
        T: Ord + Copy + Sync + Send,
    {
        for (bits, value) in tree.get(&symbol) {
            self.write(*bits, *value).await?
        }

        Ok(())
    }
}

/// For counting the number of bits written but generating no output.
///
/// # Example
/// ```
/// use bitstream_io::{BigEndian, AsyncBitWrite, BitCounter};
/// let mut writer: BitCounter<u32, BigEndian> = BitCounter::new();
/// # tokio_test::block_on(async {
/// writer.write(1, 0b1).await.unwrap();
/// writer.write(2, 0b01).await.unwrap();
/// writer.write(5, 0b10111).await.unwrap();
/// assert_eq!(writer.written(), 8);
/// # });
/// ```
#[derive(Default)]
pub struct BitCounter<N, E: AsyncEndianness> {
    bits: N,
    phantom: PhantomData<E>,
}

impl<N: Default + Copy, E: AsyncEndianness> BitCounter<N, E> {
    /// Creates new counter
    #[inline]
    pub fn new() -> Self {
        BitCounter {
            bits: N::default(),
            phantom: PhantomData,
        }
    }

    /// Returns number of bits written
    #[inline]
    pub fn written(&self) -> N {
        self.bits
    }
}

#[async_trait]
impl<N, E> AsyncBitWrite for BitCounter<N, E>
where
    E: AsyncEndianness + Sync + Send,
    N: Copy + AddAssign + From<u32> + Rem<Output = N> + PartialEq + Sync + Send,
{
    #[inline]
    async fn write_bit(&mut self, _bit: bool) -> io::Result<()> {
        self.bits += 1.into();
        Ok(())
    }

    #[inline]
    async fn write<U>(&mut self, bits: u32, value: U) -> io::Result<()>
    where
        U: Numeric + Sync + Send,
    {
        if bits > U::bits_size() {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "excessive bits for type written",
            ))
        } else if (bits < U::bits_size()) && (value >= (U::one() << bits)) {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "excessive value for bits written",
            ))
        } else {
            self.bits += bits.into();
            Ok(())
        }
    }

    #[inline]
    async fn write_signed<S>(&mut self, bits: u32, value: S) -> io::Result<()>
    where
        S: SignedNumeric + Sync + Send,
    {
        E::write_signed(self, bits, value).await
    }

    #[inline]
    async fn write_unary1(&mut self, value: u32) -> io::Result<()> {
        self.bits += (value + 1).into();
        Ok(())
    }

    #[inline]
    async fn write_unary0(&mut self, value: u32) -> io::Result<()> {
        self.bits += (value + 1).into();
        Ok(())
    }

    #[inline]
    async fn write_bytes(&mut self, buf: &[u8]) -> io::Result<()> {
        self.bits += (buf.len() as u32 * 8).into();
        Ok(())
    }

    #[inline]
    fn byte_aligned(&self) -> bool {
        self.bits % 8.into() == 0.into()
    }
}

#[async_trait]
impl<N, E> AsyncHuffmanWrite<E> for BitCounter<N, E>
where
    E: AsyncEndianness + Sync + Send,
    N: AddAssign + From<u32> + Sync + Send,
{
    async fn write_huffman<T>(&mut self, tree: &WriteHuffmanTree<E, T>, symbol: T) -> io::Result<()>
    where
        T: Ord + Copy + Sync + Send,
    {
        for &(bits, _) in tree.get(&symbol) {
            let bits: N = bits.into();
            self.bits += bits;
        }
        Ok(())
    }
}

/// A generic unsigned value for stream recording purposes
pub struct UnsignedValue(InnerUnsignedValue);

enum InnerUnsignedValue {
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    U128(u128),
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
}

macro_rules! define_unsigned_value {
    ($t:ty, $n:ident) => {
        impl From<$t> for UnsignedValue {
            #[inline]
            fn from(v: $t) -> Self {
                UnsignedValue(InnerUnsignedValue::$n(v))
            }
        }
    };
}
define_unsigned_value!(u8, U8);
define_unsigned_value!(u16, U16);
define_unsigned_value!(u32, U32);
define_unsigned_value!(u64, U64);
define_unsigned_value!(u128, U128);
define_unsigned_value!(i8, I8);
define_unsigned_value!(i16, I16);
define_unsigned_value!(i32, I32);
define_unsigned_value!(i64, I64);
define_unsigned_value!(i128, I128);

/// A generic signed value for stream recording purposes
pub struct SignedValue(InnerSignedValue);

enum InnerSignedValue {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    I128(i128),
}

macro_rules! define_signed_value {
    ($t:ty, $n:ident) => {
        impl From<$t> for SignedValue {
            #[inline]
            fn from(v: $t) -> Self {
                SignedValue(InnerSignedValue::$n(v))
            }
        }
    };
}
define_signed_value!(i8, I8);
define_signed_value!(i16, I16);
define_signed_value!(i32, I32);
define_signed_value!(i64, I64);
define_signed_value!(i128, I128);

enum WriteRecord {
    Bit(bool),
    Unsigned { bits: u32, value: UnsignedValue },
    Signed { bits: u32, value: SignedValue },
    Unary0(u32),
    Unary1(u32),
    Bytes(Box<[u8]>),
}

impl WriteRecord {
    async fn playback<W: AsyncBitWrite>(&self, writer: &mut W) -> io::Result<()> {
        match self {
            WriteRecord::Bit(v) => writer.write_bit(*v),
            WriteRecord::Unsigned {
                bits,
                value: UnsignedValue(value),
            } => match value {
                InnerUnsignedValue::U8(v) => writer.write(*bits, *v),
                InnerUnsignedValue::U16(v) => writer.write(*bits, *v),
                InnerUnsignedValue::U32(v) => writer.write(*bits, *v),
                InnerUnsignedValue::U64(v) => writer.write(*bits, *v),
                InnerUnsignedValue::U128(v) => writer.write(*bits, *v),
                InnerUnsignedValue::I8(v) => writer.write(*bits, *v),
                InnerUnsignedValue::I16(v) => writer.write(*bits, *v),
                InnerUnsignedValue::I32(v) => writer.write(*bits, *v),
                InnerUnsignedValue::I64(v) => writer.write(*bits, *v),
                InnerUnsignedValue::I128(v) => writer.write(*bits, *v),
            },
            WriteRecord::Signed {
                bits,
                value: SignedValue(value),
            } => match value {
                InnerSignedValue::I8(v) => writer.write_signed(*bits, *v),
                InnerSignedValue::I16(v) => writer.write_signed(*bits, *v),
                InnerSignedValue::I32(v) => writer.write_signed(*bits, *v),
                InnerSignedValue::I64(v) => writer.write_signed(*bits, *v),
                InnerSignedValue::I128(v) => writer.write_signed(*bits, *v),
            },
            WriteRecord::Unary0(v) => writer.write_unary0(*v),
            WriteRecord::Unary1(v) => writer.write_unary1(*v),
            WriteRecord::Bytes(bytes) => writer.write_bytes(bytes),
        }.await
    }
}

/// For recording writes in order to play them back on another writer
/// # Example
/// ```
/// use tokio::io::AsyncWrite;
/// use bitstream_io::{BigEndian, AsyncBitWriter, AsyncBitWrite, BitRecorder};
/// # tokio_test::block_on(async {
/// let mut recorder: BitRecorder<u32, BigEndian> = BitRecorder::new();
/// recorder.write(1, 0b1u64).await.unwrap();
/// recorder.write(2, 0b01u64).await.unwrap();
/// recorder.write(5, 0b10111u64).await.unwrap();
/// assert_eq!(recorder.written(), 8);
/// let mut writer = AsyncBitWriter::endian(Vec::new(), BigEndian);
/// recorder.playback(&mut writer).await;
/// assert_eq!(writer.into_writer(), [0b10110111]);
/// });
/// ```
#[derive(Default)]
pub struct BitRecorder<N, E: AsyncEndianness> {
    counter: BitCounter<N, E>,
    records: Vec<WriteRecord>,
}

impl<N: Default + Copy, E: AsyncEndianness> BitRecorder<N, E> {
    /// Creates new recorder
    #[inline]
    pub fn new() -> Self {
        BitRecorder {
            counter: BitCounter::new(),
            records: Vec::new(),
        }
    }

    /// Creates new recorder sized for the given number of writes
    #[inline]
    pub fn with_capacity(writes: usize) -> Self {
        BitRecorder {
            counter: BitCounter::new(),
            records: Vec::with_capacity(writes),
        }
    }

    /// Creates new recorder with the given endiannness
    #[inline]
    pub fn endian(_endian: E) -> Self {
        BitRecorder {
            counter: BitCounter::new(),
            records: Vec::new(),
        }
    }

    /// Returns number of bits written
    #[inline]
    pub fn written(&self) -> N {
        self.counter.written()
    }

    /// Plays recorded writes to the given writer
    #[inline]
    pub async fn playback<W: AsyncBitWrite>(&self, writer: &mut W) -> io::Result<()> {
        for record in self.records.iter() {
            record.playback(writer).await?
        }
        Ok(())
    }
}

#[async_trait]
impl<N, E> AsyncBitWrite for BitRecorder<N, E>
where
    E: AsyncEndianness + Sync + Send,
    N: Copy + From<u32> + AddAssign + Rem<Output = N> + Eq + Sync + Send,
{
    #[inline]
    async fn write_bit(&mut self, bit: bool) -> io::Result<()> {
        self.records.push(WriteRecord::Bit(bit));
        self.counter.write_bit(bit).await
    }

    #[inline]
    async fn write<U>(&mut self, bits: u32, value: U) -> io::Result<()>
    where
        U: Numeric,
    {
        self.counter.write(bits, value).await?;
        self.records.push(WriteRecord::Unsigned {
            bits,
            value: value.unsigned_value(),
        });
        Ok(())
    }

    #[inline]
    async fn write_signed<S>(&mut self, bits: u32, value: S) -> io::Result<()>
    where
        S: SignedNumeric,
    {
        self.counter.write_signed(bits, value).await?;
        self.records.push(WriteRecord::Signed {
            bits,
            value: value.signed_value(),
        });
        Ok(())
    }

    #[inline]
    async fn write_unary0(&mut self, value: u32) -> io::Result<()> {
        self.records.push(WriteRecord::Unary0(value));
        self.counter.write_unary0(value).await
    }

    #[inline]
    async fn write_unary1(&mut self, value: u32) -> io::Result<()> {
        self.records.push(WriteRecord::Unary1(value));
        self.counter.write_unary1(value).await
    }

    #[inline]
    async fn write_bytes(&mut self, buf: &[u8]) -> io::Result<()> {
        self.records.push(WriteRecord::Bytes(buf.into()));
        self.counter.write_bytes(buf).await
    }

    #[inline]
    fn byte_aligned(&self) -> bool {
        self.counter.byte_aligned()
    }
}

#[async_trait]
impl<N, E> AsyncHuffmanWrite<E> for BitRecorder<N, E>
where
    E: AsyncEndianness,
    N: Copy + From<u32> + AddAssign + Rem<Output = N> + Eq + Sync + Send,
{
    #[inline]
    async fn write_huffman<T>(&mut self, tree: &WriteHuffmanTree<E, T>, symbol: T) -> io::Result<()>
    where
        T: Ord + Copy + Sync + Send,
    {
        for (bits, value) in tree.get(&symbol) {
            self.write(*bits, *value).await?;
        }
        Ok(())
    }
}

#[inline]
async fn write_byte<W>(mut writer: W, byte: u8) -> io::Result<()>
where
    W: AsyncWrite + Unpin + Sync + Send,
{
    let buf = [byte];
    writer.write_all(&buf).await
}

async fn write_unaligned<W, E, N>(
    writer: W,
    acc: &mut BitQueue<E, N>,
    rem: &mut BitQueue<E, u8>,
) -> io::Result<()>
where
    W: AsyncWrite + Unpin + Sync + Send,
    E: AsyncEndianness + Sync + Send,
    N: Numeric + Sync + Send,
{
    if rem.is_empty() {
        Ok(())
    } else {
        use std::cmp::min;
        let bits_to_transfer = min(8 - rem.len(), acc.len());
        rem.push(bits_to_transfer, acc.pop(bits_to_transfer).to_u8());
        if rem.len() == 8 {
            write_byte(writer, rem.pop(8)).await
        } else {
            Ok(())
        }
    }
}

async fn write_aligned<W, E, N>(mut writer: W, acc: &mut BitQueue<E, N>) -> io::Result<()>
where
    W: AsyncWrite + Unpin + Sync + Send,
    E: AsyncEndianness + Sync + Send,
    N: Numeric + Sync + Send,
{
    let to_write = (acc.len() / 8) as usize;
    if to_write > 0 {
        let mut buf = N::buffer();
        let buf_ref: &mut [u8] = buf.as_mut();
        for b in buf_ref[0..to_write].iter_mut() {
            *b = acc.pop(8).to_u8();
        }
        writer.write_all(&buf_ref[0..to_write]).await
    } else {
        Ok(())
    }
}

/// For writing aligned bytes to a stream of bytes in a given endianness.
///
/// This only writes aligned values and maintains no internal state.
pub struct AsyncByteWriter<W: AsyncWrite, E: AsyncEndianness> {
    phantom: PhantomData<E>,
    writer: W,
}

impl<W: AsyncWrite + Unpin + Sync + Send, E: AsyncEndianness + Sync + Send> AsyncByteWriter<W, E> {
    /// Wraps an AsyncBitWriter around something that implements `AsyncWrite`
    pub fn new(writer: W) -> AsyncByteWriter<W, E> {
        AsyncByteWriter {
            phantom: PhantomData,
            writer,
        }
    }

    /// Wraps an AsyncBitWriter around something that implements `AsyncWrite`
    /// with the given endianness.
    pub fn endian(writer: W, _endian: E) -> AsyncByteWriter<W, E> {
        AsyncByteWriter {
            phantom: PhantomData,
            writer,
        }
    }

    /// Unwraps internal writer and disposes of `ByteWriter`.
    /// Any unwritten partial bits are discarded.
    #[inline]
    pub fn into_writer(self) -> W {
        self.writer
    }

    /// Provides mutable reference to internal writer.
    #[inline]
    pub fn writer(&mut self) -> &mut W {
        &mut self.writer
    }

    /// Converts `ByteWriter` to `BitWriter` in the same endianness.
    #[inline]
    pub fn into_bitwriter(self) -> AsyncBitWriter<W, E> {
        AsyncBitWriter::new(self.into_writer())
    }

    /// Provides temporary `BitWriter` in the same endianness.
    ///
    /// # Warning
    ///
    /// Any unwritten bits left over when `BitWriter` is dropped are lost.
    #[inline]
    pub fn bitwriter(&mut self) -> AsyncBitWriter<&mut W, E> {
        AsyncBitWriter::new(self.writer())
    }
}

/// A trait for anything that can write aligned values to an output stream
#[async_trait]
pub trait AsyncByteWrite {
    /// Writes whole numeric value to stream
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underlying stream.
    /// # Examples
    /// ```
    /// use tokio::io::AsyncWrite;
    /// use bitstream_io::{BigEndian, AsyncBitWriter, AsyncBitWrite};
    /// # tokio_test::block_on(async {
    /// let mut writer = AsyncBitWriter::endian(Vec::new(), BigEndian);
    /// writer.write(16, 0b0000000011111111u16).await.unwrap();
    /// assert_eq!(writer.into_writer(), [0b00000000, 0b11111111]);
    /// # });
    /// ```
    ///
    /// ```
    /// use tokio::io::AsyncWrite;
    /// use bitstream_io::{LittleEndian, AsyncBitWriter, AsyncBitWrite};
    /// # tokio_test::block_on(async {
    /// let mut writer = AsyncBitWriter::endian(Vec::new(), LittleEndian);
    /// writer.write(16, 0b0000000011111111u16).await.unwrap();
    /// assert_eq!(writer.into_writer(), [0b11111111, 0b00000000]);
    /// # });
    /// ```
    async fn write<N: Numeric>(&mut self, value: N) -> io::Result<()>;

    /// Writes the entirety of a byte buffer to the stream.
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underlying stream.
    async fn write_bytes(&mut self, buf: &[u8]) -> io::Result<()>;
}

#[async_trait]
impl<W: AsyncWrite + Unpin + Sync + Send, E: AsyncEndianness + Sync + Send> AsyncByteWrite for AsyncByteWriter<W, E> {
    #[inline]
    async fn write<N: Numeric>(&mut self, value: N) -> io::Result<()> {
        E::write_numeric(&mut self.writer, value).await
    }

    #[inline]
    async fn write_bytes(&mut self, buf: &[u8]) -> io::Result<()> {
        self.writer.write_all(buf).await
    }
}
