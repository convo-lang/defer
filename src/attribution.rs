//! Token attribution: which context positions were most responsible for
//! the next generated token.
//!
//! Two methods are provided:
//!
//! * **Attention × value norm** (cheap, can run live during generation).
//!   For the query that produced the next-token logits, each layer/head's
//!   attention weight to position `t` is scaled by `‖W_o,h · v_t‖`, the
//!   size of what `t` actually writes into the residual stream through that
//!   head, normalised per head, then averaged over all layers and heads.
//!   Plain mean attention is reported alongside it.
//!
//! * **Leave-one-out ablation** (faithful, expensive, run offline). Position
//!   `t` is hidden from attention for every later query in every layer, the
//!   context is recomputed, and the drop in the target token's
//!   log-probability is reported. Positions keep their RoPE indices, so the
//!   only change is the information flow from `t`. Because the engine is
//!   bit-exact, every non-zero delta is caused by the ablation alone.
//!
//! All loops are sequential and rankings break ties by lower position, so
//! attributions are as reproducible as generation itself.

use crate::error::{DeferError, Result};
use crate::model::{AttnTrace, KvCache, Model};

/// Attribution scores of one context position.
#[derive(Debug, Clone, PartialEq)]
pub struct PositionScore {
    pub position: usize,
    pub token: u32,
    /// Mean post-softmax attention over all layers and heads (sums to 1).
    pub attention: f32,
    /// Attention weighted by value-output norm, per-head normalised (sums to 1).
    pub attn_value: f32,
}

/// Result of a leave-one-out ablation of one position.
#[derive(Debug, Clone, PartialEq)]
pub struct AblationScore {
    pub position: usize,
    pub token: u32,
    /// log p(target) with this position hidden.
    pub ablated_logprob: f64,
    /// `base_logprob - ablated_logprob`. Positive: the position supported
    /// the target token. Negative: it argued against it.
    pub delta: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AblationReport {
    pub target: u32,
    /// log p(target | full context), temperature 1.
    pub base_logprob: f64,
    /// In the order the positions were requested.
    pub scores: Vec<AblationScore>,
}

/// Explanation of one generated token.
#[derive(Debug, Clone, PartialEq)]
pub struct Explanation {
    /// Index of the explained token in the session history.
    pub position: usize,
    pub token: u32,
    /// log p(token) under the raw (temperature 1) distribution.
    pub logprob: f64,
    /// Top positions, ranked by `attn_value`.
    pub top: Vec<PositionScore>,
}

/// `log softmax(logits)[idx]`, computed in f64, sequentially.
pub fn log_softmax_at(logits: &[f32], idx: usize) -> f64 {
    let mut max = f64::NEG_INFINITY;
    for &l in logits {
        if (l as f64) > max {
            max = l as f64;
        }
    }
    let mut sum = 0.0f64;
    for &l in logits {
        if !l.is_nan() {
            sum += (l as f64 - max).exp();
        }
    }
    (logits[idx] as f64 - max) - sum.ln()
}

/// Cache of `‖W_o,h · v_{t,kv(h)}‖` per (layer, position, query head).
/// Values only depend on the (deterministic) cached V vectors, so they are
/// computed once per position and reused for every later query.
#[derive(Debug, Clone, Default)]
pub struct ValueNorms {
    num_heads: usize,
    len: usize,
    /// `per_layer[l][t * num_heads + h]`
    per_layer: Vec<Vec<f32>>,
}

impl ValueNorms {
    pub fn new(model: &Model) -> Self {
        ValueNorms {
            num_heads: model.config.num_heads,
            len: 0,
            per_layer: vec![Vec::new(); model.config.num_layers],
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bring the cache in line with `cache` (extend, or drop stale entries).
    pub fn sync(&mut self, model: &Model, cache: &KvCache) {
        let c = &model.config;
        let n = cache.len();
        if n < self.len {
            for v in self.per_layer.iter_mut() {
                v.truncate(n * self.num_heads);
            }
            self.len = n;
        }
        let (hd, hidden, q_dim, kv_dim) = (c.head_dim, c.hidden_size, c.q_dim(), c.kv_dim());
        let group = c.num_heads / c.num_kv_heads;
        for l in 0..c.num_layers {
            let wo = model.attn_output_weight(l).data();
            let vals = &cache.values[l];
            for t in self.len..n {
                for h in 0..c.num_heads {
                    let kvh = h / group;
                    let v = &vals[t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                    let mut sq = 0.0f32;
                    for i in 0..hidden {
                        let row = &wo[i * q_dim + h * hd..i * q_dim + (h + 1) * hd];
                        let mut o = 0.0f32;
                        for d in 0..hd {
                            o += row[d] * v[d];
                        }
                        sq += o * o;
                    }
                    self.per_layer[l].push(sq.sqrt());
                }
            }
        }
        self.len = n;
    }

    pub fn get(&self, layer: usize, t: usize, head: usize) -> f32 {
        self.per_layer[layer][t * self.num_heads + head]
    }
}

/// Per-position scores for the query recorded in `trace`. `context` are the
/// tokens at positions `0..=trace.pos`; `norms` must cover them.
pub fn attention_attribution(trace: &AttnTrace, norms: &ValueNorms, context: &[u32]) -> Vec<PositionScore> {
    let n = trace.pos + 1;
    assert!(context.len() >= n && norms.len() >= n, "context/norms do not cover the trace");
    let layers = trace.probs.len();
    let heads = trace.num_heads;
    let mut attention = vec![0.0f32; n];
    let mut attn_value = vec![0.0f32; n];
    let mut w = vec![0.0f32; n];
    for l in 0..layers {
        for h in 0..heads {
            let mut z = 0.0f32;
            for t in 0..n {
                let a = trace.at(l, h, t);
                attention[t] += a;
                w[t] = a * norms.get(l, t, h);
                z += w[t];
            }
            if z > 0.0 {
                for t in 0..n {
                    attn_value[t] += w[t] / z;
                }
            }
        }
    }
    let denom = (layers * heads) as f32;
    (0..n)
        .map(|t| PositionScore {
            position: t,
            token: context[t],
            attention: attention[t] / denom,
            attn_value: attn_value[t] / denom,
        })
        .collect()
}

/// Sort descending by `key`, ties broken by lower position.
pub fn rank_by<T>(items: &mut [T], key: impl Fn(&T) -> f64, position: impl Fn(&T) -> usize) {
    items.sort_by(|a, b| key(b).total_cmp(&key(a)).then(position(a).cmp(&position(b))));
}

/// Top `k` positions by `attn_value`.
pub fn top_k(mut scores: Vec<PositionScore>, k: usize) -> Vec<PositionScore> {
    rank_by(&mut scores, |s| s.attn_value as f64, |s| s.position);
    scores.truncate(k);
    scores
}

/// Replay `context` and return (attention attribution, next-token logits)
/// for the query at the last context position.
pub fn attention_for_context(model: &Model, context: &[u32]) -> Result<(Vec<PositionScore>, Vec<f32>)> {
    let (last, prefix) = context
        .split_last()
        .ok_or_else(|| DeferError::Model("cannot attribute an empty context".into()))?;
    let mut cache = model.new_cache();
    for &t in prefix {
        model.forward(t, &mut cache)?;
    }
    let mut trace = AttnTrace::default();
    let logits = model.forward_traced(*last, &mut cache, &mut trace)?;
    let mut norms = ValueNorms::new(model);
    norms.sync(model, &cache);
    Ok((attention_attribution(&trace, &norms, context), logits))
}

/// Leave-one-out ablation of `positions` for predicting `target` after
/// `context`. The final context position (the query itself) can not be
/// hidden from itself and is rejected.
pub fn ablation_attribution(
    model: &Model,
    context: &[u32],
    target: u32,
    positions: &[usize],
) -> Result<AblationReport> {
    let n = context.len();
    if n == 0 {
        return Err(DeferError::Model("cannot ablate an empty context".into()));
    }
    if target as usize >= model.config.vocab_size {
        return Err(DeferError::Model(format!("target token {target} out of range")));
    }
    if let Some(&bad) = positions.iter().find(|&&p| p + 1 >= n) {
        return Err(DeferError::Model(format!(
            "position {bad} can not be ablated (must be < {}; the query token itself is excluded)",
            n - 1
        )));
    }
    let mut full = model.new_cache();
    let mut base_logits = Vec::new();
    for &t in context {
        base_logits = model.forward(t, &mut full)?;
    }
    let base_logprob = log_softmax_at(&base_logits, target as usize);

    let mut scores = Vec::with_capacity(positions.len());
    for &p in positions {
        // Positions 0..=p are unaffected by hiding p from *later* queries,
        // so the cache prefix is reused and only p+1.. is recomputed.
        let mut cache = full.clone();
        cache.truncate(p + 1);
        let mut logits = Vec::new();
        for &t in &context[p + 1..] {
            logits = model.forward_ext(t, &mut cache, &[p], None)?;
        }
        let lp = log_softmax_at(&logits, target as usize);
        scores.push(AblationScore { position: p, token: context[p], ablated_logprob: lp, delta: base_logprob - lp });
    }
    Ok(AblationReport { target, base_logprob, scores })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log_softmax() {
        let l = [1.0f32, 2.0, 3.0];
        let z: f64 = l.iter().map(|&x| (x as f64).exp()).sum();
        assert!((log_softmax_at(&l, 2) - (3.0 - z.ln())).abs() < 1e-12);
    }

    #[test]
    fn ranking_ties_prefer_lower_position() {
        let mk = |p, s| PositionScore { position: p, token: 0, attention: s, attn_value: s };
        let r = top_k(vec![mk(0, 0.1), mk(1, 0.5), mk(2, 0.5), mk(3, 0.2)], 3);
        assert_eq!(r.iter().map(|s| s.position).collect::<Vec<_>>(), vec![1, 2, 3]);
    }
}
