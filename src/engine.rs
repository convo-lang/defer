//! Generation sessions and the Checkout Engine.
//!
//! Invariant maintained by [`Session`]: every token in `history` has been
//! run through the model, so `cache.len() == history.len()` and
//! `logits` holds the next-token distribution. A checkout therefore only
//! has to replay `history` through the same code path to reproduce the
//! cache bit-for-bit.

use crate::checkpoint::DeferCheckpoint;
use crate::error::{DeferError, Result};
use crate::model::{KvCache, Model};
use crate::sampler::{sample_token, words_per_sample};
use crate::sha256::to_hex;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Expand a numeric CLI seed into the 32-byte ChaCha8 seed.
pub fn seed_from_u64(seed: u64) -> [u8; 32] {
    ChaCha8Rng::seed_from_u64(seed).get_seed()
}

pub struct Session<'m> {
    model: &'m Model,
    cache: KvCache,
    rng: ChaCha8Rng,
    base_seed: [u8; 32],
    temperature: f32,
    history: Vec<u32>,
    prompt_len: usize,
    logits: Option<Vec<f32>>,
}

impl<'m> Session<'m> {
    pub fn new(model: &'m Model, base_seed: [u8; 32], temperature: f32) -> Result<Self> {
        if !(temperature >= 0.0 && temperature.is_finite()) {
            return Err(DeferError::Model("temperature must be finite and >= 0".into()));
        }
        Ok(Session {
            model,
            cache: model.new_cache(),
            rng: ChaCha8Rng::from_seed(base_seed),
            base_seed,
            temperature,
            history: Vec::new(),
            prompt_len: 0,
            logits: None,
        })
    }

    /// Feed prompt tokens (only allowed before the first sampled token).
    pub fn prompt(&mut self, tokens: &[u32]) -> Result<()> {
        if self.history.len() != self.prompt_len {
            return Err(DeferError::Model("cannot add prompt tokens after generation started".into()));
        }
        for &t in tokens {
            self.feed(t)?;
        }
        self.prompt_len = self.history.len();
        Ok(())
    }

    fn feed(&mut self, token: u32) -> Result<()> {
        let logits = self.model.forward(token, &mut self.cache)?;
        self.history.push(token);
        self.logits = Some(logits);
        Ok(())
    }

    fn sample(&mut self) -> Result<u32> {
        let logits = self
            .logits
            .as_ref()
            .ok_or_else(|| DeferError::Model("no context: provide a non-empty prompt".into()))?;
        Ok(sample_token(logits, self.temperature, &mut self.rng) as u32)
    }

    /// Sample the next token, append it and run it through the model.
    pub fn step(&mut self) -> Result<u32> {
        let tok = self.sample()?;
        self.feed(tok)?;
        Ok(tok)
    }

    /// Generate up to `max_tokens`, stopping after any token in `stop`.
    /// `on_token` is invoked after each token is committed.
    pub fn generate(
        &mut self,
        max_tokens: usize,
        stop: &[u32],
        mut on_token: impl FnMut(&Session<'m>, u32) -> Result<()>,
    ) -> Result<Vec<u32>> {
        let mut out = Vec::new();
        for _ in 0..max_tokens {
            let t = self.step()?;
            out.push(t);
            on_token(self, t)?;
            if stop.contains(&t) {
                break;
            }
        }
        Ok(out)
    }

    pub fn history(&self) -> &[u32] {
        &self.history
    }
    pub fn prompt_len(&self) -> usize {
        self.prompt_len
    }
    pub fn generated(&self) -> &[u32] {
        &self.history[self.prompt_len..]
    }
    pub fn cache(&self) -> &KvCache {
        &self.cache
    }
    pub fn last_logits(&self) -> Option<&[f32]> {
        self.logits.as_deref()
    }
    pub fn rng_word_pos(&self) -> u128 {
        self.rng.get_word_pos()
    }
    pub fn temperature(&self) -> f32 {
        self.temperature
    }
    pub fn base_seed(&self) -> [u8; 32] {
        self.base_seed
    }

    pub fn checkpoint(&self) -> DeferCheckpoint {
        DeferCheckpoint {
            model_signature: self.model.signature,
            base_seed: self.base_seed,
            temperature: self.temperature,
            history_tokens: self.history.clone(),
            prompt_len: self.prompt_len as u32,
            rng_word_pos: self.rng_word_pos(),
        }
    }

    /// Checkout Engine: validate the checkpoint against `model` and rebuild
    /// the exact session state by fast-forward recomputation.
    ///
    /// Every generated position is re-sampled during replay and compared
    /// with the recorded token, so any non-determinism or tampering is
    /// reported as [`DeferError::Divergence`] instead of silently producing
    /// a different state.
    pub fn checkout(model: &'m Model, ckpt: &DeferCheckpoint) -> Result<Self> {
        if ckpt.model_signature != model.signature {
            return Err(DeferError::Checkpoint(format!(
                "model signature mismatch: checkpoint {} vs loaded {}",
                to_hex(&ckpt.model_signature),
                to_hex(&model.signature)
            )));
        }
        let plen = ckpt.prompt_len as usize;
        if plen > ckpt.history_tokens.len() {
            return Err(DeferError::Checkpoint("prompt_len exceeds history".into()));
        }
        let expected_rng = (ckpt.history_tokens.len() - plen) as u128 * words_per_sample(ckpt.temperature);
        if ckpt.rng_word_pos != expected_rng {
            return Err(DeferError::Checkpoint(format!(
                "rng_word_pos {} inconsistent with {} sampled tokens (expected {expected_rng})",
                ckpt.rng_word_pos,
                ckpt.history_tokens.len() - plen
            )));
        }
        let mut s = Session::new(model, ckpt.base_seed, ckpt.temperature)?;
        s.prompt(&ckpt.history_tokens[..plen])?;
        for (i, &recorded) in ckpt.history_tokens.iter().enumerate().skip(plen) {
            let sampled = s.sample()?;
            if sampled != recorded {
                return Err(DeferError::Divergence(format!(
                    "replay sampled token {sampled} at position {i}, checkpoint recorded {recorded}"
                )));
            }
            s.feed(recorded)?;
        }
        debug_assert_eq!(s.rng_word_pos(), ckpt.rng_word_pos);
        Ok(s)
    }
}
