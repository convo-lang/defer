//! Attribution must never perturb generation, and must itself be
//! deterministic and consistent between the live and offline paths.

use defer::attribution::{self, log_softmax_at};
use defer::engine::{seed_from_u64, Session};
use defer::model::Model;
use defer::ops::bit_eq;
use defer::synthetic;
use defer::tokenizer::Tokenizer;
use std::sync::OnceLock;

const PROMPT: &str = "Attribution should be reproducible.";

fn model() -> &'static Model {
    static M: OnceLock<Model> = OnceLock::new();
    M.get_or_init(|| {
        let d = std::env::temp_dir().join(format!("defer-attr-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("tiny.safetensors");
        synthetic::write_model(&p, &synthetic::tiny_config(), 99).unwrap();
        Model::load(&p).unwrap()
    })
}

fn session(trace: bool) -> Session<'static> {
    let mut s = Session::new(model(), seed_from_u64(5), 0.8).unwrap();
    if trace {
        s.enable_tracing();
    }
    s.prompt(&Tokenizer::bytes().encode(PROMPT).unwrap()).unwrap();
    s
}

#[test]
fn tracing_and_explaining_do_not_perturb_generation() {
    let mut plain = session(false);
    let a = plain.generate(30, &[], |_, _| Ok(())).unwrap();

    let mut traced = session(true);
    let b = traced.generate_explained(30, &[], 5, |_, _| Ok(())).unwrap();

    assert_eq!(a, b);
    assert!(plain.cache().bit_eq(traced.cache()));
    assert!(bit_eq(plain.last_logits().unwrap(), traced.last_logits().unwrap()));
    assert_eq!(plain.rng_word_pos(), traced.rng_word_pos());
}

#[test]
fn scores_cover_context_and_sum_to_one() {
    let mut s = session(true);
    s.generate(10, &[], |_, _| Ok(())).unwrap();
    let scores = s.explain_next().unwrap();
    assert_eq!(scores.len(), s.history().len());
    for (i, sc) in scores.iter().enumerate() {
        assert_eq!(sc.position, i);
        assert_eq!(sc.token, s.history()[i]);
        assert!(sc.attention >= 0.0 && sc.attn_value >= 0.0);
    }
    let sa: f32 = scores.iter().map(|x| x.attention).sum();
    let sv: f32 = scores.iter().map(|x| x.attn_value).sum();
    assert!((sa - 1.0).abs() < 1e-4 && (sv - 1.0).abs() < 1e-4, "{sa} {sv}");
}

#[test]
fn late_tracing_fallback_matches_live_tracing() {
    let mut live = session(true);
    live.generate(12, &[], |_, _| Ok(())).unwrap();
    let mut late = session(false);
    late.generate(12, &[], |_, _| Ok(())).unwrap();
    // Enabling tracing late recomputes the last token; state must not move.
    assert_eq!(live.explain_next().unwrap(), late.explain_next().unwrap());
    assert!(live.cache().bit_eq(late.cache()));
    assert_eq!(
        live.generate(5, &[], |_, _| Ok(())).unwrap(),
        late.generate(5, &[], |_, _| Ok(())).unwrap()
    );
}

#[test]
fn live_and_offline_attributions_are_identical() {
    let mut s = session(true);
    let mut live = Vec::new();
    s.generate_explained(8, &[], usize::MAX, |_, e| {
        live.push(e.clone());
        Ok(())
    })
    .unwrap();
    for e in &live {
        let ctx = &s.history()[..e.position];
        let (scores, logits) = attribution::attention_for_context(model(), ctx).unwrap();
        assert_eq!(attribution::top_k(scores, usize::MAX), e.top);
        assert_eq!(log_softmax_at(&logits, e.token as usize).to_bits(), e.logprob.to_bits());
    }
}

#[test]
fn ablation_is_deterministic_and_matches_full_recompute() {
    let mut s = session(false);
    s.generate(6, &[], |_, _| Ok(())).unwrap();
    let h = s.history();
    let (ctx, target) = (&h[..h.len() - 1], *h.last().unwrap());
    let positions: Vec<usize> = (0..ctx.len() - 1).collect();

    let r1 = attribution::ablation_attribution(model(), ctx, target, &positions).unwrap();
    let r2 = attribution::ablation_attribution(model(), ctx, target, &positions).unwrap();
    assert_eq!(r1, r2);
    assert!(r1.scores.iter().any(|x| x.delta != 0.0), "ablation should change something");

    // Prefix reuse must equal recomputing everything with the mask applied.
    for &p in &[0usize, ctx.len() / 2, ctx.len() - 2] {
        let mut cache = model().new_cache();
        let mut logits = Vec::new();
        for &t in ctx {
            logits = model().forward_ext(t, &mut cache, &[p], None).unwrap();
        }
        let lp = log_softmax_at(&logits, target as usize);
        assert_eq!(lp.to_bits(), r1.scores[p].ablated_logprob.to_bits(), "position {p}");
    }

    // Empty mask is bit-identical to the plain forward pass.
    let (mut c1, mut c2) = (model().new_cache(), model().new_cache());
    for &t in ctx {
        let a = model().forward(t, &mut c1).unwrap();
        let b = model().forward_ext(t, &mut c2, &[], None).unwrap();
        assert!(bit_eq(&a, &b));
    }
    // The query token itself can not be ablated.
    assert!(attribution::ablation_attribution(model(), ctx, target, &[ctx.len() - 1]).is_err());
}
