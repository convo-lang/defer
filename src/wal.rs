//! Binary write-ahead log of generated tokens.
//!
//! Each sampled token is appended (and flushed) as a fixed-size record
//! together with the PRNG word position *after* sampling it. Because the
//! engine is deterministic, this log is all that is needed to rebuild the
//! exact KV cache and PRNG state at any step.
//!
//! Layout (little-endian):
//! ```text
//! header:
//!   magic        8  b"DEFERWAL"
//!   version      4  u32 = 1
//!   signature   32  model SHA-256
//!   base_seed   32
//!   temperature  4  f32 bits
//!   prompt_len   4  u32   (original prompt length inside the prefix)
//!   start_rng   16  u128  PRNG word position at WAL start
//!   prefix_len   4  u32
//!   prefix   4 * n  u32 tokens already in context when the WAL began
//!   crc32        4  over the header bytes above
//! record (29 bytes, repeated):
//!   tag          1  0x01 = token
//!   step         4  u32 absolute index in history
//!   token        4  u32
//!   rng_pos     16  u128 PRNG word position after sampling
//!   crc32        4  over the 25 bytes above
//! ```
//! A torn (partially written) or corrupt final record is detected by its
//! CRC and discarded on read; [`WalWriter::resume`] truncates it away.

use crate::checkpoint::{Cursor, DeferCheckpoint};
use crate::crc32::crc32;
use crate::error::{DeferError, Result};
use crate::sampler::words_per_sample;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

const MAGIC: &[u8; 8] = b"DEFERWAL";
const VERSION: u32 = 1;
const TAG_TOKEN: u8 = 0x01;
pub const RECORD_SIZE: usize = 29;

#[derive(Debug, Clone, PartialEq)]
pub struct WalHeader {
    pub model_signature: [u8; 32],
    pub base_seed: [u8; 32],
    pub temperature: f32,
    pub prompt_len: u32,
    pub start_rng_word_pos: u128,
    pub prefix_tokens: Vec<u32>,
}

impl WalHeader {
    fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&VERSION.to_le_bytes());
        b.extend_from_slice(&self.model_signature);
        b.extend_from_slice(&self.base_seed);
        b.extend_from_slice(&self.temperature.to_bits().to_le_bytes());
        b.extend_from_slice(&self.prompt_len.to_le_bytes());
        b.extend_from_slice(&self.start_rng_word_pos.to_le_bytes());
        b.extend_from_slice(&(self.prefix_tokens.len() as u32).to_le_bytes());
        for t in &self.prefix_tokens {
            b.extend_from_slice(&t.to_le_bytes());
        }
        let c = crc32(&b);
        b.extend_from_slice(&c.to_le_bytes());
        b
    }

    /// Header derived from a checkpoint (continue logging after checkout).
    pub fn from_checkpoint(c: &DeferCheckpoint) -> Self {
        WalHeader {
            model_signature: c.model_signature,
            base_seed: c.base_seed,
            temperature: c.temperature,
            prompt_len: c.prompt_len,
            start_rng_word_pos: c.rng_word_pos,
            prefix_tokens: c.history_tokens.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalRecord {
    pub step: u32,
    pub token: u32,
    pub rng_word_pos: u128,
}

impl WalRecord {
    fn to_bytes(&self) -> [u8; RECORD_SIZE] {
        let mut b = [0u8; RECORD_SIZE];
        b[0] = TAG_TOKEN;
        b[1..5].copy_from_slice(&self.step.to_le_bytes());
        b[5..9].copy_from_slice(&self.token.to_le_bytes());
        b[9..25].copy_from_slice(&self.rng_word_pos.to_le_bytes());
        let c = crc32(&b[..25]);
        b[25..29].copy_from_slice(&c.to_le_bytes());
        b
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WalLog {
    pub header: WalHeader,
    pub records: Vec<WalRecord>,
    /// Byte offset just past the last valid record.
    pub valid_len: u64,
    /// True if trailing bytes were discarded (torn write / corruption).
    pub torn_tail: bool,
}

impl WalLog {
    pub fn read(path: &Path) -> Result<Self> {
        let mut bytes = Vec::new();
        File::open(path)?.read_to_end(&mut bytes)?;
        Self::parse(&bytes)
    }

    pub fn parse(b: &[u8]) -> Result<Self> {
        let err = |m: &str| DeferError::Wal(m.to_string());
        const FIXED: usize = 8 + 4 + 32 + 32 + 4 + 4 + 16 + 4;
        if b.len() < FIXED + 4 || &b[..8] != MAGIC {
            return Err(err("not a defer WAL"));
        }
        let mut r = Cursor { b, i: 8 };
        if r.u32() != VERSION {
            return Err(err("unsupported WAL version"));
        }
        let model_signature = r.arr32();
        let base_seed = r.arr32();
        let temperature = f32::from_bits(r.u32());
        let prompt_len = r.u32();
        let start_rng_word_pos = r.u128();
        let n = r.u32() as usize;
        let header_end = FIXED + 4 * n;
        if b.len() < header_end + 4 {
            return Err(err("truncated WAL header"));
        }
        let prefix_tokens = (0..n).map(|_| r.u32()).collect();
        let stored = r.u32();
        if crc32(&b[..header_end]) != stored {
            return Err(err("WAL header checksum mismatch"));
        }
        let header = WalHeader {
            model_signature,
            base_seed,
            temperature,
            prompt_len,
            start_rng_word_pos,
            prefix_tokens,
        };
        if prompt_len as usize > n {
            return Err(err("prompt_len exceeds prefix length"));
        }

        let mut records = Vec::new();
        let mut off = header_end + 4;
        let mut expect_step = n as u32;
        let mut expect_rng = start_rng_word_pos;
        let mut torn_tail = false;
        while off < b.len() {
            if b.len() - off < RECORD_SIZE {
                torn_tail = true;
                break;
            }
            let rec = &b[off..off + RECORD_SIZE];
            if rec[0] != TAG_TOKEN || crc32(&rec[..25]) != u32::from_le_bytes(rec[25..29].try_into().unwrap()) {
                torn_tail = true;
                break;
            }
            let mut c = Cursor { b: rec, i: 0 };
            c.u8();
            let r = WalRecord { step: c.u32(), token: c.u32(), rng_word_pos: c.u128() };
            expect_rng += words_per_sample(temperature);
            if r.step != expect_step || r.rng_word_pos != expect_rng {
                return Err(DeferError::Wal(format!(
                    "record {} is inconsistent (step {} / rng {}, expected {} / {})",
                    records.len(),
                    r.step,
                    r.rng_word_pos,
                    expect_step,
                    expect_rng
                )));
            }
            expect_step += 1;
            records.push(r);
            off += RECORD_SIZE;
        }
        Ok(WalLog { header, records, valid_len: off as u64, torn_tail })
    }

    /// Collapse the log into a checkpoint of the latest state.
    pub fn to_checkpoint(&self) -> DeferCheckpoint {
        let h = &self.header;
        let mut history = h.prefix_tokens.clone();
        history.extend(self.records.iter().map(|r| r.token));
        DeferCheckpoint {
            model_signature: h.model_signature,
            base_seed: h.base_seed,
            temperature: h.temperature,
            history_tokens: history,
            prompt_len: h.prompt_len,
            rng_word_pos: self.records.last().map_or(h.start_rng_word_pos, |r| r.rng_word_pos),
        }
    }
}

pub struct WalWriter {
    out: BufWriter<File>,
    next_step: u32,
    durable: bool,
}

impl WalWriter {
    /// Create (truncate) a WAL file and write its header. With `durable`,
    /// every append is fsync'd, not just flushed.
    pub fn create(path: &Path, header: &WalHeader, durable: bool) -> Result<Self> {
        let f = File::create(path)?;
        let mut out = BufWriter::new(f);
        out.write_all(&header.to_bytes())?;
        out.flush()?;
        out.get_ref().sync_all()?;
        Ok(WalWriter { out, next_step: header.prefix_tokens.len() as u32, durable })
    }

    /// Reopen an existing WAL for appending, discarding any torn tail.
    pub fn resume(path: &Path, durable: bool) -> Result<(WalLog, Self)> {
        let log = WalLog::read(path)?;
        let mut f = OpenOptions::new().write(true).open(path)?;
        f.set_len(log.valid_len)?;
        f.seek(SeekFrom::End(0))?;
        f.sync_all()?;
        let next_step = (log.header.prefix_tokens.len() + log.records.len()) as u32;
        Ok((log, WalWriter { out: BufWriter::new(f), next_step, durable }))
    }

    pub fn append(&mut self, token: u32, rng_word_pos: u128) -> Result<()> {
        let rec = WalRecord { step: self.next_step, token, rng_word_pos };
        self.out.write_all(&rec.to_bytes())?;
        self.out.flush()?;
        if self.durable {
            self.out.get_ref().sync_data()?;
        }
        self.next_step += 1;
        Ok(())
    }

    pub fn next_step(&self) -> u32 {
        self.next_step
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header() -> WalHeader {
        WalHeader {
            model_signature: [1; 32],
            base_seed: [2; 32],
            temperature: 0.5,
            prompt_len: 3,
            start_rng_word_pos: 0,
            prefix_tokens: vec![10, 11, 12],
        }
    }

    #[test]
    fn write_read_torn_resume() {
        let dir = std::env::temp_dir().join(format!("defer-wal-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.wal");
        let mut w = WalWriter::create(&p, &header(), false).unwrap();
        for (i, t) in [5u32, 6, 7].iter().enumerate() {
            w.append(*t, i as u128 + 1).unwrap();
        }
        drop(w);
        // Simulate a torn write.
        let mut f = OpenOptions::new().append(true).open(&p).unwrap();
        f.write_all(&[1, 2, 3, 4, 5]).unwrap();
        drop(f);

        let log = WalLog::read(&p).unwrap();
        assert!(log.torn_tail);
        assert_eq!(log.records.len(), 3);
        let ck = log.to_checkpoint();
        assert_eq!(ck.history_tokens, vec![10, 11, 12, 5, 6, 7]);
        assert_eq!(ck.rng_word_pos, 3);

        let (_, mut w) = WalWriter::resume(&p, true).unwrap();
        w.append(8, 4).unwrap();
        drop(w);
        let log = WalLog::read(&p).unwrap();
        assert!(!log.torn_tail);
        assert_eq!(log.records.last().unwrap().token, 8);
        std::fs::remove_dir_all(&dir).ok();
    }
}
