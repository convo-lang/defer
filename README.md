# defer

**A strictly deterministic, single-threaded LLM inference engine written in pure Rust.**

`defer` runs Llama-family language models on the CPU. It guarantees that the same weights, prompt, temperature and seed always produce the exact same tokens, byte for byte, down to the bits of every logit and KV-cache entry.

Most inference engines trade reproducibility for throughput. They use parallel GPU kernels, BLAS libraries and dynamic batching, all of which reorder floating-point additions from run to run. `defer` makes the opposite trade. Every operation runs in a fixed, sequential order, so the output is a pure function of its inputs.

That guarantee has a useful consequence: **the state of a conversation doesn't need to be stored, because it can be recomputed.** Instead of saving gigabytes of KV-cache tensors, `defer` keeps a tiny write-ahead log of token IDs. From that log it can rebuild the exact internal state of any session, verified bit for bit.

```bash
$ defer --model SmolLM2-135M/model.safetensors --prompt "The capital of France is" \
        --temperature 0.7 --seed 42 --wal session.wal
# ...any time later, on any copy of the same weights:
$ defer checkout --model SmolLM2-135M/model.safetensors --from session.wal --max-tokens 50
[defer] checked out 13 tokens (5 prompt + 8 generated), replay verified
```

---

## Why determinism?

### Advantages

| | Typical inference engine | `defer` |
|---|---|---|
| Same input, same output | Usually close, not guaranteed | Always, bit for bit |
| Persisting context | Store the KV cache (e.g. ~46 KB per token for SmolLM2-135M, ~46 MB for 1,000 tokens) | Store token IDs (~4 KB for 1,000 tokens) |
| Resuming a session | Reload tensors, or re-run and hope it matches | Replay the log, verified token by token |
| Detecting tampering or corruption | Not possible | A replayed token that differs from the record is reported as a `Divergence` error |
| Attribution and ablation experiments | Signal mixed with run-to-run noise | Every measured difference is real |
| Dependencies | CUDA, BLAS, Python, ... | `rand` + `rand_chacha` |

- **Reproducibility is something you can test for.** A prompt plus a seed is a complete specification of an output. You can store it, diff it, assert on it in CI, and get the same answer next year.
- **Context costs almost nothing to store.** A checkpoint is 112 bytes plus 4 bytes per token, and a WAL record is 29 bytes per token. That is roughly 10,000× smaller than the equivalent KV cache.
- **Every replay is an audit.** Rebuilding a session re-samples each generated position and checks it against the record. A wrong model, a modified log or any hidden non-determinism is caught immediately.
- **Crash-safe by construction.** Each token is appended to a CRC-protected log as it is generated. A partial final write is detected and discarded, and generation resumes exactly where it left off.
- **Controlled experiments.** Since nothing varies between runs, you can change one input, such as hiding a single context token, and every change in the output is caused by that edit and nothing else.

### The trade-off

`defer` is deliberately slow. It runs on a single CPU thread with naive loops, at about 17 tokens/s for SmolLM2-135M on a laptop CPU. It is meant for workloads where correctness, auditability and storage matter more than throughput. It is not a replacement for a production serving stack.

### How determinism is enforced

1. **Single-threaded execution.** There is no rayon, no threads and no async runtime in the math path. Every loop is sequential.
2. **No reassociation.** Rust never enables fast-math or FP contraction, so the loop order in the source is the order that runs. There is no BLAS or CUDA with dynamic parallel reductions.
3. **One code path.** Prompt tokens and generated tokens go through the same one-token-at-a-time forward pass. A rebuilt KV cache is therefore produced by exactly the same operations as the original.
4. **Explicit PRNG tracking.** One isolated `ChaCha8Rng` is used, and each sampled token consumes exactly one 32-bit word. Greedy decoding consumes none. The PRNG's position is logged with every token.
5. **Deterministic tie-breaking.** Exactly equal logits resolve to the lowest token index, and NaN is treated as `-inf`.
6. **No hash-order leakage.** Hash maps are only ever used for lookups, never iterated to produce output.

Results are bit-identical across runs on the same binary and platform. Transcendental functions (`exp`, `sin`, `cos`) come from the platform's math library, so bit-equality *across* operating systems or CPU architectures is not guaranteed.

---

## Example use cases

- **Reproducible research and evaluation.** Publish a benchmark result as `(model hash, prompt, seed, temperature)`. Anyone can regenerate the identical outputs, including sampled ones, without trusting your logs.
- **Golden-output regression tests.** Snapshot an LLM feature's output in CI and fail the build if a prompt, tokenizer or weights change alters even one token.
- **Audit trails and compliance.** Keep a few kilobytes per conversation and prove later exactly what the model said and why. `checkout` re-derives the transcript and refuses one that doesn't match the recorded model signature.
- **Cheap long-lived sessions.** Persist millions of chat sessions as token logs instead of KV-cache blobs, and rebuild the exact state when a user returns.
- **Branching and time travel.** Check out any point in a conversation's history and explore alternative continuations from an identical starting state.
- **Crash-safe generation jobs.** Long batch runs survive restarts. Resume from the WAL and the continuation is identical to an uninterrupted run.
- **Verifiable outputs.** A third party with the same weights can confirm that a given text really came from a given model, prompt and seed.
- **Interpretability and attribution.** Run leave-one-out ablations with zero noise floor, so every non-zero effect is real (see [Token attribution](#token-attribution)).
- **Debugging prompt pipelines.** Reproduce a bad output from a bug report exactly, then inspect it token by token.

---

## Getting started

### Prerequisites

- Rust (latest stable)

### Build and test

```bash
cargo build --release
cargo test --release        # 35 tests, including the 100-run determinism suite
```

### Try it without downloading a model

```bash
cargo run --release -- make-test-model --out tiny.safetensors
cargo run --release -- --model tiny.safetensors --prompt "hello" --max-tokens 20
```

The test model has random weights and uses a byte tokenizer, so its output is gibberish. It is still deterministic gibberish.

### Use a real model

`defer` loads Llama-family models (Llama, TinyLlama, SmolLM, Qwen2/2.5, Mistral-style) from a `.safetensors` file. It looks for `config.json` and `tokenizer.json` in the same directory.

```bash
mkdir -p models/smollm2-135m && cd models/smollm2-135m
for f in model.safetensors config.json tokenizer.json; do
  curl -L -o $f https://huggingface.co/HuggingFaceTB/SmolLM2-135M/resolve/main/$f
done
cd ../..

cargo run --release -- --model models/smollm2-135m/model.safetensors \
    --prompt "The alignment problem can be solved by" \
    --temperature 0.7 --seed 42424242 --max-tokens 64
```

Run the same command twice and you get the same text twice.

---

## Usage

```
defer [run]    --model <file> --prompt <text> [options]   generate text
defer checkout --model <file> --from <ckpt|wal> [options] rebuild a session (and continue)
defer explain  --model <file> --from <ckpt|wal> [options] attribute a recorded token
defer verify   --model <file> --prompt <text> [--runs N]  self-test determinism
defer inspect  <ckpt|wal>                                 show a log/checkpoint
defer make-test-model --out <file> [--seed N]             write a tiny random model
```

| Option | Meaning | Default |
|---|---|---|
| `--temperature <f32>` | Sampling temperature; `0` = greedy (the seed is ignored) | `0.7` |
| `--seed <u64>` / `--seed-hex <64 hex>` | PRNG seed | `0` |
| `--max-tokens <n>` | Tokens to generate | `128` (`0` for `checkout`) |
| `--wal <file>` | Append every token to a write-ahead log | |
| `--durable` | fsync the WAL after every token | |
| `--checkpoint <file>` | Write a checkpoint when finished | |
| `--tokenizer <file>` | Explicit `tokenizer.json` | next to the model |
| `--no-bos` / `--ignore-eos` | BOS/EOS handling | |
| `--explain`, `--explain-top <k>`, `--explain-jsonl <file>` | Live attribution | `k = 3` |
| `--quiet` | Only print generated text | |

### Recording, resuming and verifying

```bash
M=models/smollm2-135m/model.safetensors

# Generate while logging every token, and save a checkpoint at the end
defer --model $M --prompt "Once upon a time" --seed 7 --max-tokens 40 \
      --wal story.wal --checkpoint story.ckpt

# See what was recorded
defer inspect story.wal

# Rebuild the exact state (verified), then keep writing to the same log
defer checkout --model $M --from story.wal --wal story.wal --max-tokens 40

# Self-check: 100 identical runs plus a bit-identical KV-cache rebuild
defer verify --model $M --prompt "Hello" --runs 100
```

If you generate 40 tokens, check out, and generate 40 more, you get exactly the same 80 tokens as a single uninterrupted run.

---

## Token attribution

`defer` can show which context positions were most responsible for a generated token. There are two methods:

| Method | Cost | What it measures |
|---|---|---|
| `attention` (attention × value norm) | About 4% overhead, runs live | For the query that produced the logits, each head's attention to position *t* is scaled by ‖W_o,h·v_t‖ (how much *t* actually contributes), normalised per head, then averaged over all layers and heads. |
| `ablation` (leave-one-out) | One partial replay per position, offline | Hides position *t* from every later query in every layer, keeps the positional encodings unchanged, recomputes, and reports Δ = log p(target) − log p(target \| *t* hidden). This is the causal answer. |

```bash
# Live: top positions behind each generated token (stderr), plus JSON lines
defer --model $M --prompt "The capital of France is" --temperature 0 --max-tokens 8 \
      --explain --explain-jsonl attr.jsonl --wal fr.wal

# Offline: explain any recorded token (negative --step counts from the end)
defer explain --model $M --from fr.wal --step 6 --method ablation
```

```
explaining position 6: " capital" (generated, logprob -2.4369, p=0.0874)
context: The⟦ capital⟧⟦ of⟧⟦ France⟧ is the→ " capital"

rank    pos  token                 attn×value  attention   Δlogprob
   1      1  " capital"                0.0973     0.0455    +2.1009
   2      3  " France"                 0.1010     0.0518    +1.7136
   3      2  " of"                     0.0586     0.0350    +1.5811
   4      0  "The"                     0.3518     0.5165    -0.1960
```

The example shows why both methods exist. Attention-based scores are dominated by the first token, an "attention sink" where the model parks attention without using it. Ablation shows that token isn't actually driving the prediction. Use `--method both` to compare them side by side, `--window W` to limit ablation cost on long contexts, and `--json` for tooling. The last context token can't be hidden from its own attention, so ablation skips it.

Attribution never disturbs generation. Tracing only copies values out of the forward pass, so a session run with `--explain` produces the same tokens, KV cache and WAL as one without it. Live and offline attributions for the same token are bit-identical.

---

## Storage formats

All formats are little-endian binary with CRC-32 integrity checks. The full layouts are documented in the source.

- **Checkpoint** (`src/checkpoint.rs`): a semantic snapshot of a session.
  ```rust
  pub struct DeferCheckpoint {
      pub model_signature: [u8; 32], // SHA-256 of the weights file
      pub base_seed: [u8; 32],       // ChaCha8 seed
      pub temperature: f32,
      pub history_tokens: Vec<u32>,  // prompt + generated tokens
      pub prompt_len: u32,           // how many leading tokens were supplied, not sampled
      pub rng_word_pos: u128,        // PRNG position (cross-check)
  }
  ```
  Checkpoints are written to a temporary file and renamed, so a crash never leaves a half-written one.

- **Write-ahead log** (`src/wal.rs`): a header (model signature, seed, temperature, prefix tokens), followed by one 29-byte record per generated token: `step`, `token`, PRNG word position and CRC. A torn final record is detected and discarded, and `--wal` on `checkout` truncates it and continues appending.

The **model signature** is the SHA-256 of the entire weights file. It is computed while the file is read, in the same sequential pass, at no extra I/O cost. Checkpoints made with different weights are rejected.

---

## Project layout

| Module | Contents |
|---|---|
| `src/tensor.rs`, `src/ops.rs` | Rank 1–3 `Tensor` over a contiguous `Vec<f32>`; sequential MatMul, linear, RMSNorm, LayerNorm, SiLU, GeLU, add, multiply, softmax |
| `src/model.rs` | Llama-family decoder: RMSNorm, RoPE, grouped-query attention, gated FFN, optional QKV bias, tied or separate LM head; append-only `KvCache` |
| `src/sampler.rs` | Temperature scaling, softmax, single-pass CDF sampling, deterministic greedy argmax |
| `src/safetensors.rs` | Single-pass sequential reader (F32/F16/BF16/F64) with integrated SHA-256; writer |
| `src/tokenizer.rs` | BPE from `tokenizer.json` (byte-level GPT-2/Llama-3/Qwen style, SentencePiece with byte fallback) plus a raw-byte tokenizer |
| `src/engine.rs` | `Session` (prompt, step, generate, checkpoint) and the verified checkout engine |
| `src/wal.rs`, `src/checkpoint.rs` | Write-ahead log and checkpoint formats |
| `src/attribution.rs` | Attention × value-norm attribution and leave-one-out ablation |
| `src/json.rs`, `src/sha256.rs`, `src/crc32.rs` | Small in-crate implementations, so there are no extra dependencies |
| `src/synthetic.rs` | Deterministic tiny test model |
| `tests/verification.rs` | 100-run zero-deviation test, bit-level checkout-vs-fresh KV-cache test, resume, crash and tamper tests |
| `tests/attribution.rs` | Attribution never perturbs generation; live, offline and ablation results are bit-consistent |

The only dependencies are [`rand`](https://crates.io/crates/rand) and [`rand_chacha`](https://crates.io/crates/rand_chacha).

### Library use

```rust
use defer::{Model, Session, Tokenizer, seed_from_u64};

let model = Model::load("model.safetensors".as_ref())?;
let tok = Tokenizer::load("tokenizer.json".as_ref())?;

let mut s = Session::new(&model, seed_from_u64(42), 0.7)?;
s.prompt(&tok.encode("The capital of France is")?)?;
s.generate(32, &model.config.eos_token_ids, |_, _| Ok(()))?;

let ckpt = s.checkpoint();                      // a few bytes per token
let restored = Session::checkout(&model, &ckpt)?; // bit-identical state, verified
assert!(restored.cache().bit_eq(s.cache()));
```

---

## Limitations

- CPU only, single-threaded, and slow by design.
- Llama-family architectures in safetensors format only (single file, no GGUF, no quantised weights; F16/BF16 are converted to F32 on load).
- Bit-exactness is guaranteed on the same platform and binary, not across different CPU architectures or math libraries.
- The byte-level pre-tokenizer approximates the GPT-2 split regex. On unusual inputs, token boundaries can differ from the reference Hugging Face tokenizer, though they are always deterministic.
- Leave-one-out ablation costs O(N²) forward passes. Use `--window` on long contexts.

## License

Apache-2.0. See [LICENSE](LICENSE).
