// Copyright 2017 Brian Langenberger
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Traits and implementations for reading bits from a stream.
//!
//! ## Example
//!
//! Reading the initial STREAMINFO block from a FLAC file,
//! as documented in its
//! [specification](https://xiph.org/flac/format.html#stream).
//!
//! ```
//! use std::io::Cursor;
//! use tokio::io::{AsyncRead, AsyncReadExt};
//! use bitstream_io::{BigEndian, AsyncBitReader, AsyncBitRead};
//!
//! # tokio_test::block_on(async {
//! let flac: Vec<u8> = vec![0x66,0x4C,0x61,0x43,0x00,0x00,0x00,0x22,
//!                          0x10,0x00,0x10,0x00,0x00,0x06,0x06,0x00,
//!                          0x21,0x62,0x0A,0xC4,0x42,0xF0,0x00,0x04,
//!                          0xA6,0xCC,0xFA,0xF2,0x69,0x2F,0xFD,0xEC,
//!                          0x2D,0x5B,0x30,0x01,0x76,0xB4,0x62,0x88,
//!                          0x7D,0x92];
//!
//! let mut cursor = Cursor::new(&flac);
//! {
//!     let mut reader = AsyncBitReader::endian(&mut cursor, BigEndian);
//!
//!     // stream marker
//!     let mut file_header: [u8; 4] = [0, 0, 0, 0];
//!     reader.read_bytes(&mut file_header).await.unwrap();
//!     assert_eq!(&file_header, b"fLaC");
//!
//!     // metadata block header
//!     let last_block: bool = reader.read_bit().await.unwrap();
//!     let block_type: u8 = reader.read(7).await.unwrap();
//!     let block_size: u32 = reader.read(24).await.unwrap();
//!     assert_eq!(last_block, false);
//!     assert_eq!(block_type, 0);
//!     assert_eq!(block_size, 34);
//!
//!     // STREAMINFO block
//!     let minimum_block_size: u16 = reader.read(16).await.unwrap();
//!     let maximum_block_size: u16 = reader.read(16).await.unwrap();
//!     let minimum_frame_size: u32 = reader.read(24).await.unwrap();
//!     let maximum_frame_size: u32 = reader.read(24).await.unwrap();
//!     let sample_rate: u32 = reader.read(20).await.unwrap();
//!     let channels = reader.read::<u8>(3).await.unwrap() + 1;
//!     let bits_per_sample = reader.read::<u8>(5).await.unwrap() + 1;
//!     let total_samples: u64 = reader.read(36).await.unwrap();
//!     assert_eq!(minimum_block_size, 4096);
//!     assert_eq!(maximum_block_size, 4096);
//!     assert_eq!(minimum_frame_size, 1542);
//!     assert_eq!(maximum_frame_size, 8546);
//!     assert_eq!(sample_rate, 44100);
//!     assert_eq!(channels, 2);
//!     assert_eq!(bits_per_sample, 16);
//!     assert_eq!(total_samples, 304844);
//! }
//!
//! // STREAMINFO's MD5 sum
//!
//! // Note that the wrapped reader can be used once bitstream reading
//! // is finished at exactly the position one would expect.
//!
//! let mut md5 = [0; 16];
//! cursor.read_exact(&mut md5).await.unwrap();
//! assert_eq!(&md5,
//!     b"\xFA\xF2\x69\x2F\xFD\xEC\x2D\x5B\x30\x01\x76\xB4\x62\x88\x7D\x92");
//! });
//! ```

#![warn(missing_docs)]

use std::io;
use tokio::io::{AsyncRead, AsyncReadExt};
use async_trait::async_trait;

use super::{huffman::ReadHuffmanTree, BitQueue, AsyncEndianness, Numeric, PhantomData, SignedNumeric};

/// A trait for anything that can read a variable number of
/// potentially un-aligned values from an input stream
#[async_trait]
pub trait AsyncBitRead: Sync + Send {
    /// Reads a single bit from the stream.
    /// `true` indicates 1, `false` indicates 0
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underlying stream.
    async fn read_bit(&mut self) -> io::Result<bool>;

    /// Reads an unsigned value from the stream with
    /// the given number of bits.
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underlying stream.
    /// Also returns an error if the output type is too small
    /// to hold the requested number of bits.
    async fn read<U>(&mut self, bits: u32) -> io::Result<U>
    where
        U: Numeric + Sync + Send;

    /// Reads a twos-complement signed value from the stream with
    /// the given number of bits.
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underlying stream.
    /// Also returns an error if the output type is too small
    /// to hold the requested number of bits.
    async fn read_signed<S>(&mut self, bits: u32) -> io::Result<S>
    where
        S: SignedNumeric + Sync + Send;

    /// Skips the given number of bits in the stream.
    /// Since this method does not need an accumulator,
    /// it may be slightly faster than reading to an empty variable.
    /// In addition, since there is no accumulator,
    /// there is no upper limit on the number of bits
    /// which may be skipped.
    /// These bits are still read from the stream, however,
    /// and are never skipped via a `seek` method.
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underlying stream.
    async fn skip(&mut self, bits: u32) -> io::Result<()>;

    /// Completely fills the given buffer with whole bytes.
    /// If the stream is already byte-aligned, it will map
    /// to a faster `read_exact` call.  Otherwise it will read
    /// bytes individually in 8-bit increments.
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underlying stream.
    async fn read_bytes(&mut self, buf: &mut [u8]) -> io::Result<()> {
        for b in buf.iter_mut() {
            *b = self.read(8).await?;
        }
        Ok(())
    }

    /// Counts the number of 1 bits in the stream until the next
    /// 0 bit and returns the amount read.
    /// Because this field is variably-sized and may be large,
    /// its output is always a `u32` type.
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underlying stream.
    async fn read_unary0(&mut self) -> io::Result<u32> {
        let mut unary = 0;
        while self.read_bit().await? {
            unary += 1;
        }
        Ok(unary)
    }

    /// Counts the number of 0 bits in the stream until the next
    /// 1 bit and returns the amount read.
    /// Because this field is variably-sized and may be large,
    /// its output is always a `u32` type.
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underlying stream.
    async fn read_unary1(&mut self) -> io::Result<u32> {
        let mut unary = 0;
        while !(self.read_bit().await?) {
            unary += 1;
        }
        Ok(unary)
    }

    /// Returns true if the stream is aligned at a whole byte.
    fn byte_aligned(&self) -> bool;

    /// Throws away all unread bit values until the next whole byte.
    /// Does nothing if the stream is already aligned.
    fn byte_align(&mut self);
}

/// A trait for anything that can read Huffman codes
/// of a given endianness from an input stream
#[async_trait]
pub trait AsyncHuffmanRead<E: AsyncEndianness>: Sync + Send {
    /// Given a compiled Huffman tree, reads bits from the stream
    /// until the next symbol is encountered.
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underlying stream.
    async fn read_huffman<T>(&mut self, tree: &[ReadHuffmanTree<E, T>]) -> io::Result<T>
    where
        T: Clone + Sync + Send;
}

/// For reading non-aligned bits from a stream of bytes in a given endianness.
///
/// This will read exactly as many whole bytes needed to return
/// the requested number of bits.  It may cache up to a single partial byte
/// but no more.
pub struct AsyncBitReader<R: AsyncRead + Unpin, E: AsyncEndianness> {
    reader: R,
    bitqueue: BitQueue<E, u8>,
}

impl<R: AsyncRead + Unpin + Sync + Send, E: AsyncEndianness + Sync + Send> AsyncBitReader<R, E> {
    /// Wraps an AsyncBitReader around something that implements `AsyncRead`
    pub fn new(reader: R) -> AsyncBitReader<R, E> {
        AsyncBitReader {
            reader,
            bitqueue: BitQueue::new(),
        }
    }

    /// Wraps an AsyncBitReader around something that implements `AsyncRead`
    /// with the given endianness.
    pub fn endian(reader: R, _endian: E) -> AsyncBitReader<R, E> {
        AsyncBitReader {
            reader,
            bitqueue: BitQueue::new(),
        }
    }

    /// Unwraps internal reader and disposes of AsyncBitReader.
    ///
    /// # Warning
    ///
    /// Any unread partial bits are discarded.
    #[inline]
    pub fn into_reader(self) -> R {
        self.reader
    }

    /// If stream is byte-aligned, provides mutable reference
    /// to internal reader.  Otherwise returns `None`
    #[inline]
    pub async fn reader(&mut self) -> Option<&mut R> {
        if self.byte_aligned() {
            Some(&mut self.reader)
        } else {
            None
        }
    }

    /// Converts `AsyncBitReader` to `AsyncByteReader` in the same endianness.
    ///
    /// # Warning
    ///
    /// Any unread partial bits are discarded.
    #[inline]
    pub fn into_bytereader(self) -> AsyncByteReader<R, E> {
        AsyncByteReader::new(self.into_reader())
    }

    /// If stream is byte-aligned, provides temporary `AsyncByteReader`
    /// in the same endianness.  Otherwise returns `None`
    ///
    /// # Warning
    ///
    /// Any reader bits left over when `AsyncByteReader` is dropped are lost.
    #[inline]
    pub async fn bytereader(&mut self) -> Option<AsyncByteReader<&mut R, E>> {
        self.reader().await.map(AsyncByteReader::new)
    }

    /// Consumes reader and returns any un-read partial byte
    /// as a `(bits, value)` tuple.
    ///
    /// # Examples
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{BigEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = [0b1010_0101, 0b0101_1010];
    /// let mut reader = AsyncBitReader::endian(Cursor::new(&data), BigEndian);
    /// assert_eq!(reader.read::<u16>(9).await.unwrap(), 0b1010_0101_0);
    /// let (bits, value) = reader.into_unread();
    /// assert_eq!(bits, 7);
    /// assert_eq!(value, 0b101_1010);
    /// # });
    /// ```
    ///
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{BigEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = [0b1010_0101, 0b0101_1010];
    /// let mut reader = AsyncBitReader::endian(Cursor::new(&data), BigEndian);
    /// assert_eq!(reader.read::<u16>(8).await.unwrap(), 0b1010_0101);
    /// let (bits, value) = reader.into_unread();
    /// assert_eq!(bits, 0);
    /// assert_eq!(value, 0);
    /// # });
    /// ```
    #[inline]
    pub fn into_unread(self) -> (u32, u8) {
        (self.bitqueue.len(), self.bitqueue.value())
    }
}

#[async_trait]
impl<R: AsyncRead + Unpin + Sync + Send, E: AsyncEndianness + Sync + Send> AsyncBitRead for AsyncBitReader<R, E> {
    /// # Examples
    ///
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{BigEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = [0b10110111];
    /// let mut reader = AsyncBitReader::endian(Cursor::new(&data), BigEndian);
    /// assert_eq!(reader.read_bit().await.unwrap(), true);
    /// assert_eq!(reader.read_bit().await.unwrap(), false);
    /// assert_eq!(reader.read_bit().await.unwrap(), true);
    /// assert_eq!(reader.read_bit().await.unwrap(), true);
    /// assert_eq!(reader.read_bit().await.unwrap(), false);
    /// assert_eq!(reader.read_bit().await.unwrap(), true);
    /// assert_eq!(reader.read_bit().await.unwrap(), true);
    /// assert_eq!(reader.read_bit().await.unwrap(), true);
    /// # });
    /// ```
    ///
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{LittleEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = [0b10110111];
    /// let mut reader = AsyncBitReader::endian(Cursor::new(&data), LittleEndian);
    /// assert_eq!(reader.read_bit().await.unwrap(), true);
    /// assert_eq!(reader.read_bit().await.unwrap(), true);
    /// assert_eq!(reader.read_bit().await.unwrap(), true);
    /// assert_eq!(reader.read_bit().await.unwrap(), false);
    /// assert_eq!(reader.read_bit().await.unwrap(), true);
    /// assert_eq!(reader.read_bit().await.unwrap(), true);
    /// assert_eq!(reader.read_bit().await.unwrap(), false);
    /// assert_eq!(reader.read_bit().await.unwrap(), true);
    /// # });
    /// ```
    #[inline(always)]
    async fn read_bit(&mut self) -> io::Result<bool> {
        if self.bitqueue.is_empty() {
            self.bitqueue.set(read_byte(&mut self.reader).await?, 8);
        }
        Ok(self.bitqueue.pop(1) == 1)
    }

    /// # Examples
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{BigEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = [0b10110111];
    /// let mut reader = AsyncBitReader::endian(Cursor::new(&data), BigEndian);
    /// assert_eq!(reader.read::<u8>(1).await.unwrap(), 0b1);
    /// assert_eq!(reader.read::<u8>(2).await.unwrap(), 0b01);
    /// assert_eq!(reader.read::<u8>(5).await.unwrap(), 0b10111);
    /// # });
    /// ```
    ///
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{LittleEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = [0b10110111];
    /// let mut reader = AsyncBitReader::endian(Cursor::new(&data), LittleEndian);
    /// assert_eq!(reader.read::<u8>(1).await.unwrap(), 0b1);
    /// assert_eq!(reader.read::<u8>(2).await.unwrap(), 0b11);
    /// assert_eq!(reader.read::<u8>(5).await.unwrap(), 0b10110);
    /// # });
    /// ```
    ///
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{BigEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = [0;10];
    /// let mut reader = AsyncBitReader::endian(Cursor::new(&data), BigEndian);
    /// assert!(reader.read::<u8>(9).await.is_err());    // can't read  9 bits to u8
    /// assert!(reader.read::<u16>(17).await.is_err());  // can't read 17 bits to u16
    /// assert!(reader.read::<u32>(33).await.is_err());  // can't read 33 bits to u32
    /// assert!(reader.read::<u64>(65).await.is_err());  // can't read 65 bits to u64
    /// # });
    /// ```
    async fn read<U>(&mut self, mut bits: u32) -> io::Result<U>
    where
        U: Numeric + Sync + Send,
    {
        if bits <= U::bits_size() {
            let bitqueue_len = self.bitqueue.len();
            if bits <= bitqueue_len {
                Ok(U::from_u8(self.bitqueue.pop(bits)))
            } else {
                let mut acc =
                    BitQueue::from_value(U::from_u8(self.bitqueue.pop_all()), bitqueue_len);
                bits -= bitqueue_len;

                read_aligned(&mut self.reader, bits / 8, &mut acc).await?;
                read_unaligned(&mut self.reader, bits % 8, &mut acc, &mut self.bitqueue).await?;
                Ok(acc.value())
            }
        } else {
            Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "excessive bits for type read",
            ))
        }
    }

    /// # Examples
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{BigEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = [0b10110111];
    /// let mut reader = AsyncBitReader::endian(Cursor::new(&data), BigEndian);
    /// assert_eq!(reader.read_signed::<i8>(4).await.unwrap(), -5);
    /// assert_eq!(reader.read_signed::<i8>(4).await.unwrap(), 7);
    /// # })
    /// ```
    ///
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{LittleEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = [0b10110111];
    /// let mut reader = AsyncBitReader::endian(Cursor::new(&data), LittleEndian);
    /// assert_eq!(reader.read_signed::<i8>(4).await.unwrap(), 7);
    /// assert_eq!(reader.read_signed::<i8>(4).await.unwrap(), -5);
    /// # });
    /// ```
    ///
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{BigEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = [0;10];
    /// let mut r = AsyncBitReader::endian(Cursor::new(&data), BigEndian);
    /// assert!(r.read_signed::<i8>(9).await.is_err());   // can't read 9 bits to i8
    /// assert!(r.read_signed::<i16>(17).await.is_err()); // can't read 17 bits to i16
    /// assert!(r.read_signed::<i32>(33).await.is_err()); // can't read 33 bits to i32
    /// assert!(r.read_signed::<i64>(65).await.is_err()); // can't read 65 bits to i64
    /// # });
    /// ```
    #[inline]
    async fn read_signed<S>(&mut self, bits: u32) -> io::Result<S>
    where
        S: SignedNumeric + Sync + Send,
    {
        E::read_signed(self, bits).await
    }

    /// # Examples
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{BigEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = [0b10110111];
    /// let mut reader = AsyncBitReader::endian(Cursor::new(&data), BigEndian);
    /// assert!(reader.skip(3).await.is_ok());
    /// assert_eq!(reader.read::<u8>(5).await.unwrap(), 0b10111);
    /// # });
    /// ```
    ///
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{LittleEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = [0b10110111];
    /// let mut reader = AsyncBitReader::endian(Cursor::new(&data), LittleEndian);
    /// assert!(reader.skip(3).await.is_ok());
    /// assert_eq!(reader.read::<u8>(5).await.unwrap(), 0b10110);
    /// # });
    /// ```
    async fn skip(&mut self, mut bits: u32) -> io::Result<()> {
        use std::cmp::min;

        let to_drop = min(self.bitqueue.len(), bits);
        if to_drop != 0 {
            self.bitqueue.drop(to_drop);
            bits -= to_drop;
        }

        skip_aligned(&mut self.reader, bits / 8).await?;
        skip_unaligned(&mut self.reader, bits % 8, &mut self.bitqueue).await
    }

    /// # Example
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{BigEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = b"foobar";
    /// let mut reader = AsyncBitReader::endian(Cursor::new(data), BigEndian);
    /// assert!(reader.skip(24).await.is_ok());
    /// let mut buf = [0;3];
    /// assert!(reader.read_bytes(&mut buf).await.is_ok());
    /// assert_eq!(&buf, b"bar");
    /// # });
    /// ```
    async fn read_bytes(&mut self, buf: &mut [u8]) -> io::Result<()> {
        if self.byte_aligned() {
            self.reader.read_exact(buf).await.map(|_| ())
        } else {
            for b in buf.iter_mut() {
                *b = self.read(8).await?;
            }
            Ok(())
        }
    }

    /// # Examples
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{BigEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = [0b01110111, 0b11111110];
    /// let mut reader = AsyncBitReader::endian(Cursor::new(&data), BigEndian);
    /// assert_eq!(reader.read_unary0().await.unwrap(), 0);
    /// assert_eq!(reader.read_unary0().await.unwrap(), 3);
    /// assert_eq!(reader.read_unary0().await.unwrap(), 10);
    /// # });
    /// ```
    ///
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{LittleEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = [0b11101110, 0b01111111];
    /// let mut reader = AsyncBitReader::endian(Cursor::new(&data), LittleEndian);
    /// assert_eq!(reader.read_unary0().await.unwrap(), 0);
    /// assert_eq!(reader.read_unary0().await.unwrap(), 3);
    /// assert_eq!(reader.read_unary0().await.unwrap(), 10);
    /// # });
    /// ```
    async fn read_unary0(&mut self) -> io::Result<u32> {
        if self.bitqueue.is_empty() {
            read_aligned_unary(&mut self.reader, 0b1111_1111, &mut self.bitqueue).await
                .map(|u| u + self.bitqueue.pop_1())
        } else if self.bitqueue.all_1() {
            let base = self.bitqueue.len();
            self.bitqueue.clear();
            read_aligned_unary(&mut self.reader, 0b1111_1111, &mut self.bitqueue).await
                .map(|u| base + u + self.bitqueue.pop_1())
        } else {
            Ok(self.bitqueue.pop_1())
        }
    }

    /// # Examples
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{BigEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = [0b10001000, 0b00000001];
    /// let mut reader = AsyncBitReader::endian(Cursor::new(&data), BigEndian);
    /// assert_eq!(reader.read_unary1().await.unwrap(), 0);
    /// assert_eq!(reader.read_unary1().await.unwrap(), 3);
    /// assert_eq!(reader.read_unary1().await.unwrap(), 10);
    /// # });
    /// ```
    ///
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{LittleEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = [0b00010001, 0b10000000];
    /// let mut reader = AsyncBitReader::endian(Cursor::new(&data), LittleEndian);
    /// assert_eq!(reader.read_unary1().await.unwrap(), 0);
    /// assert_eq!(reader.read_unary1().await.unwrap(), 3);
    /// assert_eq!(reader.read_unary1().await.unwrap(), 10);
    /// # });
    /// ```
    async fn read_unary1(&mut self) -> io::Result<u32> {
        if self.bitqueue.is_empty() {
            read_aligned_unary(&mut self.reader, 0b0000_0000, &mut self.bitqueue).await
                .map(|u| u + self.bitqueue.pop_0())
        } else if self.bitqueue.all_0() {
            let base = self.bitqueue.len();
            self.bitqueue.clear();
            read_aligned_unary(&mut self.reader, 0b0000_0000, &mut self.bitqueue).await
                .map(|u| base + u + self.bitqueue.pop_0())
        } else {
            Ok(self.bitqueue.pop_0())
        }
    }

    /// # Example
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{BigEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = [0];
    /// let mut reader = AsyncBitReader::endian(Cursor::new(&data), BigEndian);
    /// assert_eq!(reader.byte_aligned(), true);
    /// assert!(reader.skip(1).await.is_ok());
    /// assert_eq!(reader.byte_aligned(), false);
    /// assert!(reader.skip(7).await.is_ok());
    /// assert_eq!(reader.byte_aligned(), true);
    /// # });
    /// ```
    #[inline]
    fn byte_aligned(&self) -> bool {
        self.bitqueue.is_empty()
    }

    /// # Example
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{BigEndian, AsyncBitReader, AsyncBitRead};
    /// # tokio_test::block_on(async {
    /// let data = [0x00, 0xFF];
    /// let mut reader = AsyncBitReader::endian(Cursor::new(&data), BigEndian);
    /// assert_eq!(reader.read::<u8>(4).await.unwrap(), 0);
    /// reader.byte_align();
    /// assert_eq!(reader.read::<u8>(8).await.unwrap(), 0xFF);
    /// # });
    /// ```
    #[inline]
    fn byte_align(&mut self) {
        self.bitqueue.clear()
    }
}

#[async_trait]
impl<R: AsyncRead + Unpin + Sync + Send, E: AsyncEndianness + Sync + Send> AsyncHuffmanRead<E> for AsyncBitReader<R, E> {
    /// # Example
    /// ```
    /// use std::io::Cursor;
    /// use bitstream_io::{BigEndian, AsyncBitReader, AsyncHuffmanRead};
    /// use bitstream_io::huffman::compile_read_tree;
    /// let tree = compile_read_tree(
    ///     vec![('a', vec![0]),
    ///          ('b', vec![1, 0]),
    ///          ('c', vec![1, 1, 0]),
    ///          ('d', vec![1, 1, 1])]).unwrap();
    /// # tokio_test::block_on(async {
    /// let data = [0b10110111];
    /// let mut reader = AsyncBitReader::endian(Cursor::new(&data), BigEndian);
    /// assert_eq!(reader.read_huffman(&tree).await.unwrap(), 'b');
    /// assert_eq!(reader.read_huffman(&tree).await.unwrap(), 'c');
    /// assert_eq!(reader.read_huffman(&tree).await.unwrap(), 'd');
    /// # });
    /// ```
    async fn read_huffman<T>(&mut self, tree: &[ReadHuffmanTree<E, T>]) -> io::Result<T>
    where
        T: Clone + Sync + Send,
    {
        let mut result: &ReadHuffmanTree<E, T> = &tree[self.bitqueue.to_state()];
        loop {
            match result {
                ReadHuffmanTree::Done(ref value, ref queue_val, ref queue_bits, _) => {
                    self.bitqueue.set(*queue_val, *queue_bits);
                    return Ok(value.clone());
                }
                ReadHuffmanTree::Continue(ref tree) => {
                    result = &tree[read_byte(&mut self.reader).await? as usize];
                }
                ReadHuffmanTree::InvalidState => {
                    panic!("invalid state");
                }
            }
        }
    }
}

#[inline]
async fn read_byte<R>(reader: &mut R) -> io::Result<u8>
where
    R: AsyncRead + Unpin + Sync + Send,
{
    let mut buf = [0; 1];
    reader.read_exact(&mut buf).await.map(|_| buf[0])
}

async fn read_aligned<R, E, N>(mut reader: R, bytes: u32, acc: &mut BitQueue<E, N>) -> io::Result<()>
where
    R: AsyncRead + Unpin + Sync + Send,
    E: AsyncEndianness + Sync + Send,
    N: Numeric,
{
    debug_assert!(bytes <= 16);

    if bytes > 0 {
        let mut buf = [0; 16];
        reader.read_exact(&mut buf[0..bytes as usize]).await?;
        for b in &buf[0..bytes as usize] {
            acc.push(8, N::from_u8(*b));
        }
    }
    Ok(())
}

async fn skip_aligned<R>(mut reader: R, mut bytes: u32) -> io::Result<()>
where
    R: AsyncRead + Unpin + Sync + Send,
{
    use std::cmp::min;

    /*skip up to 8 bytes at a time
    (unlike with read_aligned, "bytes" may be larger than any native type)*/
    let mut buf = [0; 8];
    while bytes > 0 {
        let to_read = min(8, bytes);
        reader.read_exact(&mut buf[0..to_read as usize]).await?;
        bytes -= to_read;
    }
    Ok(())
}

#[inline]
async fn read_unaligned<R, E, N>(
    mut reader: R,
    bits: u32,
    acc: &mut BitQueue<E, N>,
    rem: &mut BitQueue<E, u8>,
) -> io::Result<()>
where
    R: AsyncRead + Unpin + Sync + Send,
    E: AsyncEndianness + Sync + Send,
    N: Numeric,
{
    debug_assert!(bits <= 8);

    if bits > 0 {
        rem.set(read_byte(&mut reader).await?, 8);
        acc.push(bits, N::from_u8(rem.pop(bits)));
    }
    Ok(())
}

#[inline]
async fn skip_unaligned<R, E>(mut reader: R, bits: u32, rem: &mut BitQueue<E, u8>) -> io::Result<()>
where
    R: AsyncRead + Unpin + Sync + Send,
    E: AsyncEndianness + Sync + Send,
{
    debug_assert!(bits <= 8);

    if bits > 0 {
        rem.set(read_byte(&mut reader).await?, 8);
        rem.pop(bits);
    }
    Ok(())
}

#[inline]
async fn read_aligned_unary<R, E>(
    mut reader: R,
    continue_val: u8,
    rem: &mut BitQueue<E, u8>,
) -> io::Result<u32>
where
    R: AsyncRead + Unpin + Sync + Send,
    E: AsyncEndianness + Sync + Send,
{
    let mut acc = 0;
    let mut byte = read_byte(&mut reader).await?;
    while byte == continue_val {
        acc += 8;
        byte = read_byte(&mut reader).await?;
    }
    rem.set(byte, 8);
    Ok(acc)
}

/// A trait for anything that can read aligned values from an input stream
#[async_trait]
pub trait AsyncByteRead {
    /// Reads whole numeric value from stream
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underlying stream.
    ///
    /// # Examples
    /// ```
    /// use std::io::Cursor;
    /// use tokio::io::{AsyncRead, AsyncReadExt};
    /// use bitstream_io::{BigEndian, AsyncByteReader, AsyncByteRead};
    /// # tokio_test::block_on(async {
    /// let data = [0b00000000, 0b11111111];
    /// let mut reader = AsyncByteReader::endian(Cursor::new(&data), BigEndian);
    /// assert_eq!(reader.read::<u16>().await.unwrap(), 0b0000000011111111);
    /// # });
    /// ```
    ///
    /// ```
    /// use std::io::Cursor;
    /// use tokio::io::{AsyncRead, AsyncReadExt};
    /// use bitstream_io::{LittleEndian, AsyncByteReader, AsyncByteRead};
    /// # tokio_test::block_on(async {
    /// let data = [0b00000000, 0b11111111];
    /// let mut reader = AsyncByteReader::endian(Cursor::new(&data), LittleEndian);
    /// assert_eq!(reader.read::<u16>().await.unwrap(), 0b1111111100000000);
    /// # });
    /// ```
    async fn read<N: Numeric + Sync + Send>(&mut self) -> Result<N, io::Error>;

    /// Completely fills the given buffer with whole bytes.
    ///
    /// # Errors
    ///
    /// Passes along any I/O error from the underlying stream.
    async fn read_bytes(&mut self, buf: &mut [u8]) -> io::Result<()> {
        for b in buf.iter_mut() {
            *b = self.read().await?;
        }
        Ok(())
    }
}

/// For reading aligned bytes from a stream of bytes in a given endianness.
///
/// This only reads aligned values and maintains no internal state.
pub struct AsyncByteReader<R: AsyncRead + Unpin, E: AsyncEndianness> {
    phantom: PhantomData<E>,
    reader: R,
}

impl<R: AsyncRead + Unpin + Sync + Send, E: AsyncEndianness + Sync + Send> AsyncByteReader<R, E> {
    /// Wraps an AsyncByteReader around something that implements `AsyncRead`
    pub fn new(reader: R) -> AsyncByteReader<R, E> {
        AsyncByteReader {
            phantom: PhantomData,
            reader,
        }
    }

    /// Wraps an AsyncByteReader around something that implements `AsyncRead`
    /// with the given endianness.
    pub fn endian(reader: R, _endian: E) -> AsyncByteReader<R, E> {
        AsyncByteReader {
            phantom: PhantomData,
            reader,
        }
    }

    /// Unwraps internal reader and disposes of `ByteReader`.
    #[inline]
    pub fn into_reader(self) -> R {
        self.reader
    }

    /// Provides mutable reference to internal reader
    #[inline]
    pub fn reader(&mut self) -> &mut R {
        &mut self.reader
    }

    /// Converts `ByteReader` to `BitReader` in the same endianness.
    #[inline]
    pub fn into_bitreader(self) -> AsyncBitReader<R, E> {
        AsyncBitReader::new(self.into_reader())
    }

    /// Provides temporary `BitReader` in the same endianness.
    ///
    /// # Warning
    ///
    /// Any unread bits left over when `BitReader` is dropped are lost.
    #[inline]
    pub fn bitreader(&mut self) -> AsyncBitReader<&mut R, E> {
        AsyncBitReader::new(self.reader())
    }
}

#[async_trait]
impl<R: AsyncRead + Unpin + Sync + Send, E: AsyncEndianness + Sync + Send> AsyncByteRead for AsyncByteReader<R, E> {
    #[inline]
    async fn read<N: Numeric + Sync + Send>(&mut self) -> Result<N, io::Error> {
        E::read_numeric(&mut self.reader).await
    }

    #[inline]
    async fn read_bytes(&mut self, buf: &mut [u8]) -> io::Result<()> {
        self.reader.read_exact(buf).await.map(|_| ())
    }
}
