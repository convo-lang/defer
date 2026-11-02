//! Phase 6 verification suite.

use defer::checkpoint::DeferCheckpoint;
use defer::engine::{seed_from_u64, Session};
use defer::error::DeferError;
use defer::model::Model;
use defer::synthetic;
use defer::tokenizer::Tokenizer;
use defer::wal::{WalHeader, WalLog, WalWriter};
use std::path::PathBuf;
use std::sync::OnceLock;

const PROMPT: &str = "The alignment problem can be solved by";
const TEMPERATURE: f32 = 0.7;
const SEED: u64 = 42424242;

fn test_dir() -> PathBuf {
    let d = std::env::temp_dir().join(format!("defer-it-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn model_path() -> &'static PathBuf {
    static P: OnceLock<PathBuf> = OnceLock::new();
    P.get_or_init(|| {
        let p = test_dir().join("tiny.safetensors");
        synthetic::write_model(&p, &synthetic::tiny_config(), 1234).unwrap();
        p
    })
}

fn model() -> &'static Model {
    static M: OnceLock<Model> = OnceLock::new();
    M.get_or_init(|| Model::load(model_path()).unwrap())
}

fn generate(model: &Model, temperature: f32, seed: u64, n: usize) -> (String, Session<'_>) {
    let tok = Tokenizer::bytes();
    let mut s = Session::new(model, seed_from_u64(seed), temperature).unwrap();
    s.prompt(&tok.encode(PROMPT).unwrap()).unwrap();
    let out = s.generate(n, &[], |_, _| Ok(())).unwrap();
    (tok.decode(&out, false), s)
}

#[test]
fn hundred_runs_zero_deviation() {
    let model = model();
    let (reference, ref_session) = generate(model, TEMPERATURE, SEED, 48);
    let ref_fp = ref_session.cache().fingerprint();
    let ref_logits = ref_session.last_logits().unwrap().to_vec();
    assert!(!reference.is_empty());
    for run in 1..100 {
        let (text, s) = generate(model, TEMPERATURE, SEED, 48);
        assert_eq!(text, reference, "run {run} deviated");
        assert_eq!(s.cache().fingerprint(), ref_fp, "run {run}: KV cache bits deviated");
        assert!(defer::ops::bit_eq(s.last_logits().unwrap(), &ref_logits));
        assert_eq!(s.rng_word_pos(), 48, "exactly one PRNG draw per token");
    }
}

#[test]
fn reloading_weights_is_deterministic() {
    let (reference, _) = generate(model(), TEMPERATURE, SEED, 32);
    for _ in 0..3 {
        let m = Model::load(model_path()).unwrap();
        assert_eq!(m.signature, model().signature);
        assert_eq!(generate(&m, TEMPERATURE, SEED, 32).0, reference);
    }
}

#[test]
fn seeds_and_temperatures_matter() {
    let m = model();
    let a = generate(m, TEMPERATURE, SEED, 32).0;
    let b = generate(m, TEMPERATURE, SEED + 1, 32).0;
    assert_ne!(a, b, "different seeds should (overwhelmingly likely) diverge");
    let g1 = generate(m, 0.0, 1, 32);
    let g2 = generate(m, 0.0, 2, 32);
    assert_eq!(g1.0, g2.0, "greedy decoding must ignore the seed");
    assert_eq!(g1.1.rng_word_pos(), 0, "greedy decoding must not touch the PRNG");
}

#[test]
fn checkout_kv_cache_is_bit_identical() {
    let m = model();
    for &temp in &[0.0f32, TEMPERATURE, 1.5] {
        let (_, live) = generate(m, temp, SEED, 40);
        let ckpt = live.checkpoint();

        // Round-trip through disk.
        let p = test_dir().join(format!("t{temp}.ckpt"));
        ckpt.save(&p).unwrap();
        let loaded = DeferCheckpoint::load(&p).unwrap();
        assert_eq!(loaded, ckpt);

        // Recomputed cache via checkout.
        let restored = Session::checkout(m, &loaded).unwrap();
        // Freshly evaluated cache: feed the whole history as a prompt.
        let mut fresh = Session::new(m, ckpt.base_seed, temp).unwrap();
        fresh.prompt(&ckpt.history_tokens).unwrap();

        assert!(restored.cache().bit_eq(live.cache()), "checkout != live (T={temp})");
        assert!(restored.cache().bit_eq(fresh.cache()), "checkout != fresh prompt eval (T={temp})");
        assert!(defer::ops::bit_eq(restored.last_logits().unwrap(), live.last_logits().unwrap()));
        assert_eq!(restored.rng_word_pos(), live.rng_word_pos());
    }
}

#[test]
fn resumed_generation_matches_uninterrupted() {
    let m = model();
    let (full, _) = generate(m, TEMPERATURE, SEED, 60);
    let (_, partial) = generate(m, TEMPERATURE, SEED, 25);
    let mut resumed = Session::checkout(m, &partial.checkpoint()).unwrap();
    resumed.generate(35, &[], |_, _| Ok(())).unwrap();
    assert_eq!(Tokenizer::bytes().decode(resumed.generated(), false), full);
}

#[test]
fn checkout_rejects_wrong_model_and_tampering() {
    let m = model();
    let (_, s) = generate(m, TEMPERATURE, SEED, 10);
    let mut bad = s.checkpoint();
    bad.model_signature[0] ^= 1;
    assert!(matches!(Session::checkout(m, &bad), Err(DeferError::Checkpoint(_))));

    let mut tampered = s.checkpoint();
    let last = tampered.history_tokens.len() - 1;
    tampered.history_tokens[last] = (tampered.history_tokens[last] + 1) % 256;
    assert!(matches!(Session::checkout(m, &tampered), Err(DeferError::Divergence(_))));

    let mut bad_rng = s.checkpoint();
    bad_rng.rng_word_pos += 1;
    assert!(Session::checkout(m, &bad_rng).is_err());
}

#[test]
fn wal_reconstructs_session() {
    let m = model();
    let tok = Tokenizer::bytes();
    let prompt = tok.encode(PROMPT).unwrap();
    let seed = seed_from_u64(SEED);
    let path = test_dir().join("gen.wal");

    let mut s = Session::new(m, seed, TEMPERATURE).unwrap();
    s.prompt(&prompt).unwrap();
    let header = WalHeader {
        model_signature: m.signature,
        base_seed: seed,
        temperature: TEMPERATURE,
        prompt_len: prompt.len() as u32,
        start_rng_word_pos: 0,
        prefix_tokens: prompt.clone(),
    };
    let mut w = WalWriter::create(&path, &header, false).unwrap();
    s.generate(30, &[], |sess, t| w.append(t, sess.rng_word_pos())).unwrap();
    drop(w);

    // Simulate a crash mid-record.
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.extend_from_slice(&[0x01, 0xAA, 0xBB]);
    std::fs::write(&path, &bytes).unwrap();

    let log = WalLog::read(&path).unwrap();
    assert!(log.torn_tail);
    assert_eq!(log.records.len(), 30);
    let ckpt = log.to_checkpoint();
    assert_eq!(ckpt, s.checkpoint());
    let restored = Session::checkout(m, &ckpt).unwrap();
    assert!(restored.cache().bit_eq(s.cache()));

    // Resume appending to the same WAL and confirm continuity.
    let (_, mut w) = WalWriter::resume(&path, false).unwrap();
    let mut cont = restored;
    cont.generate(5, &[], |sess, t| w.append(t, sess.rng_word_pos())).unwrap();
    drop(w);
    let log = WalLog::read(&path).unwrap();
    assert!(!log.torn_tail);
    assert_eq!(log.to_checkpoint(), cont.checkpoint());
}
