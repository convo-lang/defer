//! Llama-family decoder-only transformer (RMSNorm, RoPE, grouped-query
//! attention, gated FFN) evaluated one token at a time.
//!
//! Prompt prefill and incremental decoding use the *same* single-token code
//! path, so a KV cache built by feeding a prompt and a KV cache rebuilt
//! from a checkpoint are produced by identical sequences of float ops and
//! are therefore bit-identical.

use crate::error::{DeferError, Result};
use crate::json::{self, Value};
use crate::ops;
use crate::safetensors::SafeTensors;
use crate::sha256::Sha256;
use crate::tensor::Tensor;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Activation {
    Silu,
    Gelu,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub max_seq_len: usize,
    pub tie_word_embeddings: bool,
    pub activation: Activation,
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Vec<u32>,
}

impl Config {
    /// Build a config from a HuggingFace-style `config.json` value. Missing
    /// sizes are inferred from the weight shapes where possible.
    pub fn from_json(v: &Value, st: &SafeTensors) -> Result<Self> {
        let num = |k: &str| v.get(k).and_then(Value::as_u64).map(|x| x as usize);
        let flt = |k: &str| v.get(k).and_then(Value::as_f64);
        let missing = |k: &str| DeferError::Model(format!("config is missing `{k}`"));

        let embed = find(st, &["model.embed_tokens.weight", "embed_tokens.weight"])
            .and_then(|n| st.get(n));
        let (vocab_guess, hidden_guess) = match embed {
            Some(t) if t.rank() == 2 => (Some(t.shape()[0]), Some(t.shape()[1])),
            _ => (None, None),
        };
        let hidden_size = num("hidden_size").or(hidden_guess).ok_or_else(|| missing("hidden_size"))?;
        let vocab_size = num("vocab_size").or(vocab_guess).ok_or_else(|| missing("vocab_size"))?;
        let num_layers = match num("num_hidden_layers") {
            Some(n) => n,
            None => {
                let mut n = 0;
                while st.contains(&format!("model.layers.{n}.input_layernorm.weight")) {
                    n += 1;
                }
                if n == 0 {
                    return Err(missing("num_hidden_layers"));
                }
                n
            }
        };
        let intermediate_size = match num("intermediate_size") {
            Some(n) => n,
            None => st
                .get("model.layers.0.mlp.gate_proj.weight")
                .map(|t| t.shape()[0])
                .ok_or_else(|| missing("intermediate_size"))?,
        };
        let num_heads = num("num_attention_heads").ok_or_else(|| missing("num_attention_heads"))?;
        let num_kv_heads = num("num_key_value_heads").unwrap_or(num_heads);
        let head_dim = num("head_dim").unwrap_or(hidden_size / num_heads);
        let activation = match v.get("hidden_act").and_then(Value::as_str).unwrap_or("silu") {
            "silu" | "swish" => Activation::Silu,
            a if a.starts_with("gelu") => Activation::Gelu,
            a => return Err(DeferError::Model(format!("unsupported activation {a}"))),
        };
        let eos_token_ids = match v.get("eos_token_id") {
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_u64).map(|x| x as u32).collect(),
            Some(x) => x.as_u64().map(|x| vec![x as u32]).unwrap_or_default(),
            None => vec![],
        };
        let has_lm_head = st.contains("lm_head.weight");
        let cfg = Config {
            vocab_size,
            hidden_size,
            intermediate_size,
            num_layers,
            num_heads,
            num_kv_heads,
            head_dim,
            rms_norm_eps: flt("rms_norm_eps").unwrap_or(1e-6) as f32,
            rope_theta: flt("rope_theta").unwrap_or(10000.0) as f32,
            max_seq_len: num("max_position_embeddings").unwrap_or(4096),
            tie_word_embeddings: v
                .get("tie_word_embeddings")
                .and_then(Value::as_bool)
                .unwrap_or(!has_lm_head)
                || !has_lm_head,
            activation,
            bos_token_id: v.get("bos_token_id").and_then(Value::as_u64).map(|x| x as u32),
            eos_token_ids,
        };
        if cfg.num_kv_heads == 0 || cfg.num_heads % cfg.num_kv_heads != 0 {
            return Err(DeferError::Model("num_attention_heads must be a multiple of num_key_value_heads".into()));
        }
        if cfg.head_dim % 2 != 0 {
            return Err(DeferError::Model("head_dim must be even for RoPE".into()));
        }
        Ok(cfg)
    }

    pub fn kv_dim(&self) -> usize {
        self.num_kv_heads * self.head_dim
    }

    pub fn q_dim(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Serialise as a HuggingFace-compatible config.json document.
    pub fn to_json(&self) -> String {
        let mut s = format!(
            "{{\"architectures\":[\"LlamaForCausalLM\"],\"vocab_size\":{},\"hidden_size\":{},\
             \"intermediate_size\":{},\"num_hidden_layers\":{},\"num_attention_heads\":{},\
             \"num_key_value_heads\":{},\"head_dim\":{},\"rms_norm_eps\":{:e},\"rope_theta\":{},\
             \"max_position_embeddings\":{},\"tie_word_embeddings\":{},\"hidden_act\":\"{}\"",
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_layers,
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.rms_norm_eps,
            self.rope_theta,
            self.max_seq_len,
            self.tie_word_embeddings,
            match self.activation {
                Activation::Silu => "silu",
                Activation::Gelu => "gelu_pytorch_tanh",
            }
        );
        if let Some(b) = self.bos_token_id {
            s.push_str(&format!(",\"bos_token_id\":{b}"));
        }
        if !self.eos_token_ids.is_empty() {
            let ids: Vec<String> = self.eos_token_ids.iter().map(|x| x.to_string()).collect();
            s.push_str(&format!(",\"eos_token_id\":[{}]", ids.join(",")));
        }
        s.push('}');
        s
    }
}

fn find<'a>(st: &SafeTensors, candidates: &[&'a str]) -> Option<&'a str> {
    candidates.iter().copied().find(|n| st.contains(n))
}

pub struct Layer {
    attn_norm: Tensor,
    wq: Tensor,
    bq: Option<Tensor>,
    wk: Tensor,
    bk: Option<Tensor>,
    wv: Tensor,
    bv: Option<Tensor>,
    wo: Tensor,
    ffn_norm: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
}

/// Append-only, strictly sequential key/value cache.
/// `keys[layer]` is a flat `[len, kv_dim]` row-major buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct KvCache {
    pub kv_dim: usize,
    pub keys: Vec<Vec<f32>>,
    pub values: Vec<Vec<f32>>,
    len: usize,
}

impl KvCache {
    pub fn new(config: &Config) -> Self {
        KvCache {
            kv_dim: config.kv_dim(),
            keys: vec![Vec::new(); config.num_layers],
            values: vec![Vec::new(); config.num_layers],
            len: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Drop every position at or after `n`.
    pub fn truncate(&mut self, n: usize) {
        if n >= self.len {
            return;
        }
        for l in 0..self.keys.len() {
            self.keys[l].truncate(n * self.kv_dim);
            self.values[l].truncate(n * self.kv_dim);
        }
        self.len = n;
    }

    /// Bit-for-bit equality of every cached key and value.
    pub fn bit_eq(&self, other: &KvCache) -> bool {
        self.len == other.len
            && self.kv_dim == other.kv_dim
            && self.keys.len() == other.keys.len()
            && (0..self.keys.len()).all(|l| {
                ops::bit_eq(&self.keys[l], &other.keys[l]) && ops::bit_eq(&self.values[l], &other.values[l])
            })
    }

    /// SHA-256 over the raw bits of the cache (layer-major, keys then values).
    pub fn fingerprint(&self) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(&(self.len as u64).to_le_bytes());
        for l in 0..self.keys.len() {
            for v in self.keys[l].iter().chain(self.values[l].iter()) {
                h.update(&v.to_bits().to_le_bytes());
            }
        }
        h.finalize()
    }
}

pub struct Model {
    pub config: Config,
    /// SHA-256 of the weights file.
    pub signature: [u8; 32],
    embed: Tensor,
    layers: Vec<Layer>,
    norm: Tensor,
    lm_head: Option<Tensor>,
    inv_freq: Vec<f32>,
}

impl Model {
    /// Load a model from a `.safetensors` file. The config is taken from the
    /// `defer.config` metadata entry if present, else from `config.json`
    /// in the same directory.
    pub fn load(path: &Path) -> Result<Self> {
        let st = SafeTensors::load(path)?;
        let cfg_value = if let Some(s) = st.meta("defer.config") {
            json::parse(s)?
        } else {
            let p = path.with_file_name("config.json");
            match std::fs::read_to_string(&p) {
                Ok(s) => json::parse(&s)?,
                Err(_) => Value::Object(vec![]),
            }
        };
        let config = Config::from_json(&cfg_value, &st)?;
        Model::from_safetensors(config, st)
    }

    pub fn from_safetensors(config: Config, mut st: SafeTensors) -> Result<Self> {
        let c = &config;
        let embed_name = if st.contains("model.embed_tokens.weight") {
            "model.embed_tokens.weight"
        } else {
            "embed_tokens.weight"
        };
        let has_bias = st.contains("model.layers.0.self_attn.q_proj.bias");
        let mut take = |name: &str, shape: &[usize]| -> Result<Tensor> {
            let t = st
                .take(name)
                .ok_or_else(|| DeferError::Model(format!("missing tensor {name}")))?;
            let n: usize = shape.iter().product();
            if t.len() != n {
                return Err(DeferError::Model(format!(
                    "tensor {name} has shape {:?}, expected {shape:?}",
                    t.shape()
                )));
            }
            t.reshape(shape)
        };
        let embed = take(embed_name, &[c.vocab_size, c.hidden_size])?;
        let mut layers = Vec::with_capacity(c.num_layers);
        let (h, q, kv, ff) = (c.hidden_size, c.q_dim(), c.kv_dim(), c.intermediate_size);
        for i in 0..c.num_layers {
            let p = format!("model.layers.{i}");
            layers.push(Layer {
                attn_norm: take(&format!("{p}.input_layernorm.weight"), &[h])?,
                wq: take(&format!("{p}.self_attn.q_proj.weight"), &[q, h])?,
                bq: if has_bias { Some(take(&format!("{p}.self_attn.q_proj.bias"), &[q])?) } else { None },
                wk: take(&format!("{p}.self_attn.k_proj.weight"), &[kv, h])?,
                bk: if has_bias { Some(take(&format!("{p}.self_attn.k_proj.bias"), &[kv])?) } else { None },
                wv: take(&format!("{p}.self_attn.v_proj.weight"), &[kv, h])?,
                bv: if has_bias { Some(take(&format!("{p}.self_attn.v_proj.bias"), &[kv])?) } else { None },
                wo: take(&format!("{p}.self_attn.o_proj.weight"), &[h, q])?,
                ffn_norm: take(&format!("{p}.post_attention_layernorm.weight"), &[h])?,
                w_gate: take(&format!("{p}.mlp.gate_proj.weight"), &[ff, h])?,
                w_up: take(&format!("{p}.mlp.up_proj.weight"), &[ff, h])?,
                w_down: take(&format!("{p}.mlp.down_proj.weight"), &[h, ff])?,
            });
        }
        let norm = take("model.norm.weight", &[h])?;
        let lm_head = if c.tie_word_embeddings {
            None
        } else {
            Some(take("lm_head.weight", &[c.vocab_size, h])?)
        };
        let half = c.head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half);
        for i in 0..half {
            let exponent = (2 * i) as f64 / c.head_dim as f64;
            inv_freq.push((1.0 / (c.rope_theta as f64).powf(exponent)) as f32);
        }
        Ok(Model {
            signature: st.signature,
            config,
            embed,
            layers,
            norm,
            lm_head,
            inv_freq,
        })
    }

    pub fn new_cache(&self) -> KvCache {
        KvCache::new(&self.config)
    }

    /// Run one token through the network at position `cache.len()`,
    /// appending its keys/values to the cache. Returns next-token logits.
    pub fn forward(&self, token: u32, cache: &mut KvCache) -> Result<Vec<f32>> {
        let c = &self.config;
        let pos = cache.len();
        if token as usize >= c.vocab_size {
            return Err(DeferError::Model(format!("token id {token} out of range")));
        }
        if pos >= c.max_seq_len {
            return Err(DeferError::Model(format!("context length {} exceeded", c.max_seq_len)));
        }
        let (hd, half) = (c.head_dim, c.head_dim / 2);
        let group = c.num_heads / c.num_kv_heads;
        let kv_dim = c.kv_dim();
        let scale = 1.0 / (hd as f32).sqrt();

        let mut x = self.embed.row(token as usize).to_vec();
        let mut xb = vec![0.0f32; c.hidden_size];
        let mut q = vec![0.0f32; c.q_dim()];
        let mut k = vec![0.0f32; kv_dim];
        let mut v = vec![0.0f32; kv_dim];
        let mut att_out = vec![0.0f32; c.q_dim()];
        let mut scores = vec![0.0f32; pos + 1];
        let mut gate = vec![0.0f32; c.intermediate_size];
        let mut up = vec![0.0f32; c.intermediate_size];

        // RoPE tables for this position.
        let mut cos = vec![0.0f32; half];
        let mut sin = vec![0.0f32; half];
        for i in 0..half {
            let angle = pos as f32 * self.inv_freq[i];
            cos[i] = angle.cos();
            sin[i] = angle.sin();
        }
        let rope = |vec: &mut [f32], n_heads: usize| {
            for h in 0..n_heads {
                let base = h * hd;
                for i in 0..half {
                    let x0 = vec[base + i];
                    let x1 = vec[base + i + half];
                    vec[base + i] = x0 * cos[i] - x1 * sin[i];
                    vec[base + i + half] = x1 * cos[i] + x0 * sin[i];
                }
            }
        };

        for (l, layer) in self.layers.iter().enumerate() {
            // --- attention ---
            ops::rmsnorm(&mut xb, &x, layer.attn_norm.data(), c.rms_norm_eps);
            ops::linear(&mut q, &xb, layer.wq.data(), layer.bq.as_ref().map(|t| t.data()));
            ops::linear(&mut k, &xb, layer.wk.data(), layer.bk.as_ref().map(|t| t.data()));
            ops::linear(&mut v, &xb, layer.wv.data(), layer.bv.as_ref().map(|t| t.data()));
            rope(&mut q, c.num_heads);
            rope(&mut k, c.num_kv_heads);
            cache.keys[l].extend_from_slice(&k);
            cache.values[l].extend_from_slice(&v);
            let keys = &cache.keys[l];
            let vals = &cache.values[l];

            for h in 0..c.num_heads {
                let kvh = h / group;
                let qh = &q[h * hd..(h + 1) * hd];
                for t in 0..=pos {
                    let kt = &keys[t * kv_dim + kvh * hd..t * kv_dim + (kvh + 1) * hd];
                    scores[t] = ops::dot(qh, kt) * scale;
                }
                ops::softmax_inplace(&mut scores);
                let out = &mut att_out[h * hd..(h + 1) * hd];
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for t in 0..=pos {
                        acc += scores[t] * vals[t * kv_dim + kvh * hd + d];
                    }
                    out[d] = acc;
                }
            }
            ops::linear(&mut xb, &att_out, layer.wo.data(), None);
            ops::add_inplace(&mut x, &xb);

            // --- feed-forward ---
            ops::rmsnorm(&mut xb, &x, layer.ffn_norm.data(), c.rms_norm_eps);
            ops::linear(&mut gate, &xb, layer.w_gate.data(), None);
            ops::linear(&mut up, &xb, layer.w_up.data(), None);
            match c.activation {
                Activation::Silu => ops::silu_inplace(&mut gate),
                Activation::Gelu => ops::gelu_inplace(&mut gate),
            }
            ops::mul_inplace(&mut gate, &up);
            ops::linear(&mut xb, &gate, layer.w_down.data(), None);
            ops::add_inplace(&mut x, &xb);
        }
        cache.len += 1;

        ops::rmsnorm(&mut xb, &x, self.norm.data(), c.rms_norm_eps);
        let head = self.lm_head.as_ref().unwrap_or(&self.embed);
        let mut logits = vec![0.0f32; c.vocab_size];
        ops::linear(&mut logits, &xb, head.data(), None);
        Ok(logits)
    }
}
