//! Semantic checkpoints: the minimal metadata needed to deterministically
//! rebuild a session (KV cache + PRNG state) by recomputation.
//!
//! Binary layout (little-endian):
//! ```text
//! magic        8  b"DEFERCKP"
//! version      4  u32 = 1
//! signature   32  SHA-256 of the model weights file
//! base_seed   32  ChaCha8 seed
//! temperature  4  f32 bits
//! prompt_len   4  u32
//! rng_pos     16  u128 ChaCha8 word position after the last sample
//! n_tokens     8  u64
//! tokens   4 * n  u32 each
//! crc32        4  over everything above
//! ```

use crate::crc32::crc32;
use crate::error::{DeferError, Result};
use std::path::Path;

const MAGIC: &[u8; 8] = b"DEFERCKP";
const VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq)]
pub struct DeferCheckpoint {
    pub model_signature: [u8; 32],
    pub base_seed: [u8; 32],
    pub temperature: f32,
    pub history_tokens: Vec<u32>,
    /// How many leading `history_tokens` were supplied (not sampled).
    pub prompt_len: u32,
    /// PRNG word position after the last sampled token. Redundant with
    /// `history_tokens.len() - prompt_len` under the one-draw-per-token
    /// rule; stored explicitly so replay can be cross-checked.
    pub rng_word_pos: u128,
}

impl DeferCheckpoint {
    pub fn generated_tokens(&self) -> &[u32] {
        &self.history_tokens[self.prompt_len as usize..]
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut b = Vec::with_capacity(112 + 4 * self.history_tokens.len());
        b.extend_from_slice(MAGIC);
        b.extend_from_slice(&VERSION.to_le_bytes());
        b.extend_from_slice(&self.model_signature);
        b.extend_from_slice(&self.base_seed);
        b.extend_from_slice(&self.temperature.to_bits().to_le_bytes());
        b.extend_from_slice(&self.prompt_len.to_le_bytes());
        b.extend_from_slice(&self.rng_word_pos.to_le_bytes());
        b.extend_from_slice(&(self.history_tokens.len() as u64).to_le_bytes());
        for t in &self.history_tokens {
            b.extend_from_slice(&t.to_le_bytes());
        }
        let c = crc32(&b);
        b.extend_from_slice(&c.to_le_bytes());
        b
    }

    pub fn from_bytes(b: &[u8]) -> Result<Self> {
        let err = |m: &str| DeferError::Checkpoint(m.to_string());
        const FIXED: usize = 8 + 4 + 32 + 32 + 4 + 4 + 16 + 8;
        if b.len() < FIXED + 4 || &b[..8] != MAGIC {
            return Err(err("not a defer checkpoint"));
        }
        let (body, tail) = b.split_at(b.len() - 4);
        if crc32(body) != u32::from_le_bytes(tail.try_into().unwrap()) {
            return Err(err("checksum mismatch (file corrupt)"));
        }
        let mut r = Cursor { b: body, i: 8 };
        if r.u32() != VERSION {
            return Err(err("unsupported checkpoint version"));
        }
        let model_signature = r.arr32();
        let base_seed = r.arr32();
        let temperature = f32::from_bits(r.u32());
        let prompt_len = r.u32();
        let rng_word_pos = r.u128();
        let n = r.u64() as usize;
        if body.len() != FIXED + 4 * n {
            return Err(err("length mismatch"));
        }
        let history_tokens = (0..n).map(|_| r.u32()).collect::<Vec<_>>();
        if prompt_len as usize > n {
            return Err(err("prompt_len exceeds history length"));
        }
        Ok(DeferCheckpoint { model_signature, base_seed, temperature, history_tokens, prompt_len, rng_word_pos })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        // Write-then-rename so a crash never leaves a half-written file.
        let tmp = path.with_extension("ckpt.tmp");
        {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&self.to_bytes())?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    pub fn load(path: &Path) -> Result<Self> {
        Self::from_bytes(&std::fs::read(path)?)
    }
}

pub(crate) struct Cursor<'a> {
    pub b: &'a [u8],
    pub i: usize,
}

impl<'a> Cursor<'a> {
    pub fn take(&mut self, n: usize) -> &'a [u8] {
        let s = &self.b[self.i..self.i + n];
        self.i += n;
        s
    }
    pub fn u8(&mut self) -> u8 {
        self.take(1)[0]
    }
    pub fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    pub fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    pub fn u128(&mut self) -> u128 {
        u128::from_le_bytes(self.take(16).try_into().unwrap())
    }
    pub fn arr32(&mut self) -> [u8; 32] {
        self.take(32).try_into().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_and_corruption() {
        let c = DeferCheckpoint {
            model_signature: [7; 32],
            base_seed: [9; 32],
            temperature: 0.7,
            history_tokens: vec![1, 2, 3, 40000],
            prompt_len: 2,
            rng_word_pos: 2,
        };
        let mut b = c.to_bytes();
        assert_eq!(DeferCheckpoint::from_bytes(&b).unwrap(), c);
        b[50] ^= 1;
        assert!(DeferCheckpoint::from_bytes(&b).is_err());
    }
}
