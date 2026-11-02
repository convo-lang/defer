//! Deterministically generated tiny Llama-style model, used by the test
//! suite and by `defer make-test-model` so the engine can be exercised
//! without downloading real weights. Uses the byte tokenizer (vocab 256).

use crate::error::Result;
use crate::model::{Activation, Config};
use crate::safetensors;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::path::Path;

pub fn tiny_config() -> Config {
    Config {
        vocab_size: 256,
        hidden_size: 64,
        intermediate_size: 160,
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        max_seq_len: 1024,
        tie_word_embeddings: false,
        activation: Activation::Silu,
        bos_token_id: None,
        eos_token_ids: vec![],
    }
}

/// Write a randomly initialised model for `config` to `path`.
pub fn write_model(path: &Path, config: &Config, seed: u64) -> Result<()> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut mat = |rows: usize, cols: usize, scale: f32| -> Vec<f32> {
        (0..rows * cols).map(|_| rng.gen_range(-1.0f32..1.0) * scale).collect()
    };
    let c = config;
    let (h, q, kv, ff, v) = (c.hidden_size, c.q_dim(), c.kv_dim(), c.intermediate_size, c.vocab_size);
    let s = |fan_in: usize| 1.7 / (fan_in as f32).sqrt();
    let mut t: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    t.push(("model.embed_tokens.weight".into(), vec![v, h], mat(v, h, 1.0)));
    for i in 0..c.num_layers {
        let p = format!("model.layers.{i}");
        t.push((format!("{p}.input_layernorm.weight"), vec![h], vec![1.0; h]));
        t.push((format!("{p}.self_attn.q_proj.weight"), vec![q, h], mat(q, h, s(h))));
        t.push((format!("{p}.self_attn.k_proj.weight"), vec![kv, h], mat(kv, h, s(h))));
        t.push((format!("{p}.self_attn.v_proj.weight"), vec![kv, h], mat(kv, h, s(h))));
        t.push((format!("{p}.self_attn.o_proj.weight"), vec![h, q], mat(h, q, s(q))));
        t.push((format!("{p}.post_attention_layernorm.weight"), vec![h], vec![1.0; h]));
        t.push((format!("{p}.mlp.gate_proj.weight"), vec![ff, h], mat(ff, h, s(h))));
        t.push((format!("{p}.mlp.up_proj.weight"), vec![ff, h], mat(ff, h, s(h))));
        t.push((format!("{p}.mlp.down_proj.weight"), vec![h, ff], mat(h, ff, s(ff))));
    }
    t.push(("model.norm.weight".into(), vec![h], vec![1.0; h]));
    if !c.tie_word_embeddings {
        t.push(("lm_head.weight".into(), vec![v, h], mat(v, h, 3.0 * s(h))));
    }
    safetensors::write(path, &t, &[("defer.config".into(), c.to_json())])
}
