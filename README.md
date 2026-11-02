### defer-inference

A strictly deterministic, single-threaded LLM inference engine written in pure Rust. 

Unlike standard inference engines that prioritize raw token throughput via highly parallelized, non-associative hardware operations, this engine prioritizes absolute reproducibility. Given the same input parameters, model weights, prompt, and PRNG seed, defer will always produce the exact same byte-for-byte token sequence. 

By optimizing for 100% mathematical reproducibility, defer eliminates the need to save massive raw floating-point tensor matrices to disk for context caching. Instead, it relies on a compute-driven, ultra-lightweight Write-Ahead Log (WAL) to reconstruct identical context states on demand. 

### Core Architectural Constraints

To maintain 100% determinism, the codebase must adhere to the following design constraints: 

1. **Single-Threaded Execution:** Do not introduce rayon, multi-threading, or asynchronous runtimes for tensor operations. All loops must be strictly sequential.
2. **Explicit PRNG Tracking:** All non-zero temperature sampling must consume exactly one state mutation from a single, isolated pseudo-random number generator (ChaCha8Rng) per token.
3. **No External BLAS/CUDA:** Avoid linking against external system BLAS or hardware-accelerated libraries that employ dynamic parallel reductions.
4. **Deterministic Tie-Breaking:** In the event of exact floating-point equality during token selection, the engine must fall back to a strict index-based priority selection.

### Project Structure & Roadmap

The development of this single-crate binary is divided into six logical implementation phases: 

### Phase 1: Tensor Core & Sequential Math

* [x] Implement a basic Tensor abstraction supporting 1D, 2D, and 3D shapes.
* [x] Write naive, single-threaded nested loop implementations for: 

  * Matrix multiplication (MatMul)
  * Layer Normalization (LayerNorm / RMSNorm)
  * Element-wise operations (SiLU, GeLU, Add, Multiply)
* [x] Ensure memory layout is contiguous in a flat Vec<f32> to prevent layout divergence.

### Phase 2: The Transformer Architecture

* [x] Implement the multi-head attention mechanism.
* [x] Build a predictable, sequential Key-Value (KV) cache structure.
* [x] Write the feed-forward network forward pass.
* [x] Assemble the network layer cascade (iterating sequentially through layer blocks).

### Phase 3: Deterministic Sampler Block

* [x] Integrate rand_chacha::ChaCha8Rng.
* [x] Implement sequential temperature scaling over logits.
* [x] Implement sequential Softmax normalization.
* [x] Implement the single-pass Cumulative Distribution Function (CDF) sampling algorithm.

### Phase 4: Model Loading & GGUF/Safetensors Parsing

* [x] Create a parser to read model weights sequentially from disk.
* [x] Set up tokenization mapping using a simple byte-pair encoding (BPE) or unigram decoder.

### Phase 5: Deterministic Storage Layer & WAL

* [x] Implement a binary Write-Ahead Log (WalWriter) that appends generated tokens sequentially alongside the internal PRNG state counter.
* [x] Build a semantic Checkpoint serialization system to capture minimal execution metadata: 

rust

pub struct DeferCheckpoint {
    pub model_signature: [u8; 32],
    pub base_seed: [u8; 32],
    pub temperature: f32,
    pub history_tokens: Vec<u32>,
}

* [x] Implement a Checkout Engine that takes a checkpoint file, validates the model weights, and deterministically reconstructs the exact historical KV cache state via fast-forward recomputation.

### Phase 6: Verification Suite

* [x] Create an automated integration test that runs inference 100 consecutive times on a fixed seed to verify zero string deviation.
* [x] Create a checkpoint checkout test confirming that a recomputed KV cache exactly matches a freshly evaluated prompt sequence down to the bit level.

### Step-by-Step Implementation Guide

### 1. Setting Up the Sample Loop

When writing the sampler block in src/sampler.rs, always pass the ChaCha8Rng instance by mutable reference. Ensure that no branching paths result in multiple random numbers being pulled for a single token generation step. 

rust

use rand::Rng;
use rand_chacha::ChaCha8Rng;

pub fn sample_token(logits: &[f32], temperature: f32, rng: &mut ChaCha8Rng) -> usize {
    if temperature == 0.0 {
        // Pure Greedy Decoding
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
    }

    let mut probs = vec![0.0; logits.len()];
    
    // Scale sequentially
    for i in 0..logits.len() {
        probs[i] = (logits[i] / temperature).exp();
    }
    
    // Normalize sequentially
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    
    // Mutate PRNG state exactly once
    let r: f32 = rng.gen();
    
    // Scan cumulative distribution
    let mut cumulative_sum = 0.0;
    for (token_id, &p) in probs.iter().enumerate() {
        cumulative_sum += p;
        if r <= cumulative_sum {
            return token_id;
        }
    }
    
    logits.len() - 1
}


### 2. Ensuring Strict Order of Operations

When computing matrix multiplications, avoid any compiler optimizations or auto-vectorization techniques that might split accumulators across unaligned boundaries if they alter evaluation order. Stick to explicit loop orders: 

rust

pub fn matmul_sequential(out: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize, k: usize) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}


### Getting Started

### Prerequisites

* Rust (Latest Stable Edition)

### Dependencies

Add the following minimal dependencies to your Cargo.toml: 

toml

[dependencies]
rand = "0.8"
rand_chacha = "0.3"


### Running the Engine

To execute text generation with a predictable outcome: 

bash

cargo run -- --model path/to/model.safetensors \
             --prompt "The alignment problem can be solved by" \
             --temperature 0.7 \
             --seed 42424242


### Implementation Status

All six phases are implemented. Module map:

| Phase | Module | Contents |
|---|---|---|
| 1 | `src/tensor.rs`, `src/ops.rs` | Rank 1–3 `Tensor` over a flat `Vec<f32>`; sequential MatMul, linear, RMSNorm, LayerNorm, SiLU, GeLU, Add, Mul, softmax |
| 2 | `src/model.rs` | Llama-family decoder (RMSNorm, RoPE, grouped-query attention, gated FFN, optional QKV bias), append-only `KvCache` |
| 3 | `src/sampler.rs` | Temperature, softmax, single-pass CDF. Exactly one ChaCha8 word per sampled token, none for greedy. Ties go to the lowest index, NaN is treated as `-inf` |
| 4 | `src/safetensors.rs`, `src/tokenizer.rs`, `src/json.rs` | Single-pass sequential safetensors reader (F32/F16/BF16/F64) that hashes the file (SHA-256 = model signature) while it reads; BPE tokenizer from `tokenizer.json` (byte-level and SentencePiece/byte-fallback), plus a raw-byte fallback |
| 5 | `src/wal.rs`, `src/checkpoint.rs`, `src/engine.rs` | CRC-protected WAL that survives torn tails, `DeferCheckpoint` binary format, `Session::checkout` Checkout Engine |
| 6 | `tests/verification.rs` | 100-run zero-deviation test; bit-level checkout-vs-fresh KV cache test; resume, WAL-crash and tamper tests |

The only dependencies are `rand` and `rand_chacha`. JSON, SHA-256, CRC-32 and f16/bf16 conversion are implemented in-crate.

#### Notes on the design

* **One code path for prefill and decode.** Prompt tokens and generated tokens go through the same `Model::forward(token, cache)` call, one at a time. So a cache rebuilt from a checkpoint is produced by exactly the same sequence of float operations as the live cache, and is bit-identical to it.
* **Verified checkout.** During fast-forward recomputation, the Checkout Engine re-samples every generated position and compares it with the recorded token. Any tampering or non-determinism is reported as `DeferError::Divergence`, so it never silently yields a different state. It also rejects checkpoints whose model signature does not match the loaded weights.
* **`DeferCheckpoint` extra fields.** The checkpoint adds `prompt_len` and `rng_word_pos` to the fields in the spec above. `prompt_len` is needed to know which positions were sampled. `rng_word_pos` is redundant under the one-draw-per-token rule, and is kept as a cross-check.
* **No reassociation.** Rust never enables fast-math or FP contraction, so the loop order in the source is the order that runs. `ops::linear` computes four *independent* output rows per pass for speed. Each row keeps a single in-order accumulator, and a unit test asserts the results are bit-identical to the naive loop.
* **Greedy tie-breaking.** The `max_by` snippet in the guide above returns the *last* maximum on ties. `sampler::argmax` uses a strict `>` instead, so the lowest index wins, as constraint 4 requires.
* **Scope of reproducibility.** Results are byte-identical across runs on the same binary and platform. `exp`, `sin` and `cos` come from the platform libm, so cross-platform bit-equality is not guaranteed.
* **Tokenizer.** The byte-level pre-tokenizer approximates the GPT-2 split regex. On unusual inputs, token boundaries may differ from the reference HF tokenizer, but they are always deterministic.

### Usage

```bash
# Build and run the test suite
cargo test --release

# Try it without downloading anything (random weights, byte tokenizer)
cargo run --release -- make-test-model --out tiny.safetensors

# Real model: a Llama-family safetensors file with config.json and tokenizer.json next to it
# (e.g. HuggingFaceTB/SmolLM2-135M, TinyLlama, Qwen2.5)
cargo run --release -- --model path/to/model.safetensors \
             --prompt "The alignment problem can be solved by" \
             --temperature 0.7 --seed 42424242 --max-tokens 64 \
             --wal run.wal --checkpoint run.ckpt

# Inspect a WAL or checkpoint
cargo run --release -- inspect run.wal

# Rebuild the exact state from a WAL/checkpoint, then keep generating (appends to the same WAL)
cargo run --release -- checkout --model path/to/model.safetensors --from run.wal --wal run.wal --max-tokens 32

# Self-check: N identical runs plus a bit-identical checkout
cargo run --release -- verify --model path/to/model.safetensors --prompt "Hello" --runs 100
```

Run `defer --help` for every option (`--seed-hex`, `--durable` fsync-per-token WAL, `--no-bos`, `--ignore-eos`, `--tokenizer`, `--quiet`).

The WAL and checkpoint binary layouts are documented at the top of `src/wal.rs` and `src/checkpoint.rs`.
