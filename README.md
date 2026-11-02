### defer-inference

A strictly deterministic, single-threaded LLM inference engine written in pure Rust. 

Unlike standard inference engines that prioritize raw token throughput via highly parallelized, non-associative hardware operations, this engine prioritizes absolute reproducibility. Given the same input parameters, model weights, prompt, and PRNG seed, defer will always produce the exact same byte-for-byte token sequence. 

### Core Architectural Constraints

To maintain 100% determinism, the codebase must adhere to the following design constraints: 

1. **Single-Threaded Execution:** Do not introduce rayon, multi-threading, or asynchronous runtimes for tensor operations. All loops must be strictly sequential.
2. **Explicit PRNG Tracking:** All non-zero temperature sampling must consume exactly one state mutation from a single, isolated pseudo-random number generator (ChaCha8Rng) per token.
3. **No External BLAS/CUDA:** Avoid linking against external system BLAS or hardware-accelerated libraries that employ dynamic parallel reductions.
4. **Deterministic Tie-Breaking:** In the event of exact floating-point equality during token selection, the engine must fall back to a strict index-based priority selection.

### Project Structure & Roadmap

The development of this single-crate binary is divided into five logical implementation phases: 

### Phase 1: Tensor Core & Sequential Math

* [ ] Implement a basic Tensor abstraction supporting 1D, 2D, and 3D shapes.
* [ ] Write naive, single-threaded nested loop implementations for: 

  * Matrix multiplication (MatMul)
  * Layer Normalization (LayerNorm / RMSNorm)
  * Element-wise operations (SiLU, GeLU, Add, Multiply)
* [ ] Ensure memory layout is contiguous in a flat Vec<f32> to prevent layout divergence.

### Phase 2: The Transformer Architecture

* [ ] Implement the multi-head attention mechanism.
* [ ] Build a predictable, sequential Key-Value (KV) cache structure.
* [ ] Write the feed-forward network forward pass.
* [ ] Assemble the network layer cascade (iterating sequentially through layer blocks).

### Phase 3: Deterministic Sampler Block

* [ ] Integrate rand_chacha::ChaCha8Rng.
* [ ] Implement sequential temperature scaling over logits.
* [ ] Implement sequential Softmax normalization.
* [ ] Implement the single-pass Cumulative Distribution Function (CDF) sampling algorithm.

### Phase 4: Model Loading & GGUF/Safetensors Parsing

* [ ] Create a parser to read model weights sequentially from disk.
* [ ] Set up tokenization mapping using a simple byte-pair encoding (BPE) or unigram decoder.

### Phase 5: Verification Suite

* [ ] Create an automated integration test that runs inference 100 consecutive times on a fixed seed to verify zero string deviation.

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

