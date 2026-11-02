//! Deterministic token sampler.
//!
//! Invariants:
//! * `temperature == 0.0` -> greedy argmax, the PRNG is **never** touched.
//! * `temperature > 0.0`  -> exactly one `u32` is drawn from the PRNG
//!   (`ChaCha8Rng::get_word_pos()` advances by exactly 1), regardless of
//!   the logits' contents. There is no branch that draws twice or skips.
//! * Exact float ties are broken in favour of the lowest token index.
//! * NaN logits are treated as `-inf` (never selected unless all are NaN).

use rand::Rng;
use rand_chacha::ChaCha8Rng;

/// Number of PRNG words consumed by one call to [`sample_token`].
pub fn words_per_sample(temperature: f32) -> u128 {
    if temperature == 0.0 {
        0
    } else {
        1
    }
}

/// Greedy selection with strict lowest-index tie-breaking.
pub fn argmax(logits: &[f32]) -> usize {
    let mut best = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    let mut found = false;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_nan() {
            continue;
        }
        // Strict `>`: an equal value later in the vocabulary never wins.
        if !found || v > best_val {
            best = i;
            best_val = v;
            found = true;
        }
    }
    best
}

/// Sequential temperature scaling + stable softmax into `probs`.
pub fn softmax_with_temperature(logits: &[f32], temperature: f32, probs: &mut Vec<f32>) {
    probs.clear();
    probs.resize(logits.len(), 0.0);
    let mut max = f32::NEG_INFINITY;
    for &l in logits {
        let s = l / temperature;
        if s > max {
            max = s;
        }
    }
    let mut sum = 0.0f32;
    for i in 0..logits.len() {
        let s = logits[i] / temperature;
        let p = if s.is_nan() { 0.0 } else { (s - max).exp() };
        probs[i] = p;
        sum += p;
    }
    if !(sum > 0.0) || !sum.is_finite() {
        // Degenerate (all -inf / NaN / overflow): fall back to one-hot argmax.
        let best = argmax(logits);
        for p in probs.iter_mut() {
            *p = 0.0;
        }
        if !probs.is_empty() {
            probs[best] = 1.0;
        }
        return;
    }
    let inv = 1.0 / sum;
    for p in probs.iter_mut() {
        *p *= inv;
    }
}

/// Sample one token id. See module docs for the determinism contract.
pub fn sample_token(logits: &[f32], temperature: f32, rng: &mut ChaCha8Rng) -> usize {
    assert!(!logits.is_empty(), "cannot sample from empty logits");
    assert!(
        temperature >= 0.0 && temperature.is_finite(),
        "temperature must be finite and >= 0"
    );
    if temperature == 0.0 {
        return argmax(logits);
    }

    let mut probs = Vec::new();
    softmax_with_temperature(logits, temperature, &mut probs);

    // Mutate PRNG state exactly once (one u32 word), unconditionally.
    let r: f32 = rng.gen();

    // Single-pass CDF scan.
    let mut cumulative = 0.0f32;
    let mut last_nonzero = 0usize;
    for (token_id, &p) in probs.iter().enumerate() {
        if p > 0.0 {
            last_nonzero = token_id;
            cumulative += p;
            if r < cumulative {
                return token_id;
            }
        }
    }
    // Rounding left the CDF total slightly below r: pick the last token
    // with non-zero mass (deterministic, never a zero-probability token).
    last_nonzero
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn greedy_ties_pick_lowest_index() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[f32::NAN, 0.0, 0.0]), 1);
        assert_eq!(argmax(&[f32::NEG_INFINITY; 3]), 0);
    }

    #[test]
    fn greedy_does_not_touch_rng() {
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        for _ in 0..10 {
            sample_token(&[0.1, 0.2, 0.3], 0.0, &mut rng);
        }
        assert_eq!(rng.get_word_pos(), 0);
    }

    #[test]
    fn exactly_one_word_per_sample() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let cases: [&[f32]; 4] = [
            &[0.0, 1.0, 2.0],
            &[f32::NEG_INFINITY, f32::NEG_INFINITY],
            &[f32::NAN, 5.0],
            &[1e30, -1e30, 1e30],
        ];
        for (i, l) in cases.iter().enumerate() {
            sample_token(l, 0.7, &mut rng);
            assert_eq!(rng.get_word_pos(), (i + 1) as u128);
        }
    }

    #[test]
    fn reproducible_and_distributed() {
        let logits = [0.0f32, 1.0, 2.0, 3.0];
        let run = || {
            let mut rng = ChaCha8Rng::seed_from_u64(42);
            (0..2000).map(|_| sample_token(&logits, 1.0, &mut rng)).collect::<Vec<_>>()
        };
        let a = run();
        assert_eq!(a, run());
        let mut counts = [0usize; 4];
        for &t in &a {
            counts[t] += 1;
        }
        assert!(counts[3] > counts[2] && counts[2] > counts[1] && counts[1] > counts[0]);
        assert!(counts[0] > 0);
    }
}
