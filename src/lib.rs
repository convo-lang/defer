//! # defer
//!
//! A strictly deterministic, single-threaded LLM inference engine.
//!
//! Given identical model weights, prompt, temperature and seed, `defer`
//! produces the exact same token sequence byte-for-byte. That property is
//! what allows the engine to persist context as a tiny write-ahead log of
//! token ids (plus PRNG position) instead of raw KV-cache tensors: any
//! historical state can be rebuilt bit-exactly by recomputation.
//!
//! ## Determinism rules enforced throughout the crate
//!
//! 1. Single-threaded: no rayon, threads or async runtimes. Every loop is
//!    sequential with a fixed iteration order.
//! 2. Exactly one PRNG state mutation (one `u32` draw from `ChaCha8Rng`)
//!    per sampled token at non-zero temperature, zero at temperature 0.
//! 3. No BLAS / CUDA / SIMD intrinsics. Rust never enables fast-math, so
//!    LLVM cannot reassociate float reductions; the written loop order is
//!    the evaluated order.
//! 4. Ties during token selection are broken by lowest token index.
//! 5. `HashMap`s are only ever used for lookups, never iterated to produce
//!    output, so their randomised ordering can not leak into results.

pub mod checkpoint;
pub mod crc32;
pub mod engine;
pub mod error;
pub mod json;
pub mod model;
pub mod ops;
pub mod safetensors;
pub mod sampler;
pub mod sha256;
pub mod synthetic;
pub mod tensor;
pub mod tokenizer;
pub mod wal;

pub use checkpoint::DeferCheckpoint;
pub use engine::{seed_from_u64, Session};
pub use error::{DeferError, Result};
pub use model::{Config, KvCache, Model};
pub use tokenizer::Tokenizer;
