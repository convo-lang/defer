use defer::attribution::{self, Explanation, PositionScore};
use defer::checkpoint::DeferCheckpoint;
use defer::json;
use defer::engine::{seed_from_u64, Session};
use defer::error::{DeferError, Result};
use defer::model::Model;
use defer::sha256::{from_hex, to_hex};
use defer::tokenizer::Tokenizer;
use defer::wal::{WalHeader, WalLog, WalWriter};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

const USAGE: &str = "\
defer - deterministic single-threaded LLM inference

USAGE:
  defer [run] --model <file.safetensors> --prompt <text> [options]
  defer checkout --model <file.safetensors> --from <file.ckpt|file.wal> [options]
  defer verify --model <file.safetensors> --prompt <text> [--runs N] [options]
  defer inspect <file.ckpt|file.wal>
  defer explain --model <file.safetensors> --from <file.ckpt|file.wal> [options]
  defer make-test-model --out <file.safetensors> [--seed N]

GENERATION OPTIONS:
  --temperature <f32>    sampling temperature, 0 = greedy        [default: 0.7]
  --seed <u64>           PRNG seed                               [default: 0]
  --seed-hex <64 hex>    raw 32-byte ChaCha8 seed (overrides --seed)
  --max-tokens <n>       tokens to generate                      [default: 128]
  --tokenizer <file>     tokenizer.json (default: next to the model)
  --no-bos               do not prepend the BOS token
  --ignore-eos           keep generating past EOS tokens
  --wal <file>           append every token to a write-ahead log
  --durable              fsync the WAL after every token
  --checkpoint <file>    write a checkpoint when generation finishes
  --quiet                only print generated text

ATTRIBUTION (run / checkout):
  --explain              print the top context positions behind each token (stderr)
  --explain-top <k>      positions per token                     [default: 3]
  --explain-jsonl <file> write per-token attributions as JSON lines

EXPLAIN OPTIONS:
  --step <n>             history index of the token to explain; negative counts
                         from the end                            [default: -1]
  --method <m>           attention | ablation | both             [default: attention]
  --window <w>           only ablate the last w positions before the query
  --top <k>              rows to show                            [default: 10]
  --json                 machine-readable output
";

struct Args {
    cmd: String,
    flags: BTreeMap<String, String>,
    positional: Vec<String>,
}

const BOOL_FLAGS: &[&str] = &["no-bos", "ignore-eos", "durable", "quiet", "help", "explain", "json"];

impl Args {
    fn parse() -> Result<Args> {
        let mut raw: Vec<String> = std::env::args().skip(1).collect();
        let cmd = match raw.first() {
            Some(c) if !c.starts_with('-') => raw.remove(0),
            _ => "run".to_string(),
        };
        let mut flags = BTreeMap::new();
        let mut positional = Vec::new();
        let mut it = raw.into_iter();
        while let Some(a) = it.next() {
            if let Some(name) = a.strip_prefix("--") {
                let (name, inline) = match name.split_once('=') {
                    Some((n, v)) => (n.to_string(), Some(v.to_string())),
                    None => (name.to_string(), None),
                };
                let val = if let Some(v) = inline {
                    v
                } else if BOOL_FLAGS.contains(&name.as_str()) {
                    "true".into()
                } else {
                    it.next().ok_or_else(|| DeferError::Parse(format!("--{name} needs a value")))?
                };
                flags.insert(name, val);
            } else if a == "-h" {
                flags.insert("help".into(), "true".into());
            } else {
                positional.push(a);
            }
        }
        Ok(Args { cmd, flags, positional })
    }

    fn get(&self, k: &str) -> Option<&str> {
        self.flags.get(k).map(String::as_str)
    }
    fn has(&self, k: &str) -> bool {
        self.flags.contains_key(k)
    }
    fn req(&self, k: &str) -> Result<&str> {
        self.get(k).ok_or_else(|| DeferError::Parse(format!("missing required --{k}")))
    }
    fn num<T: std::str::FromStr>(&self, k: &str, default: T) -> Result<T> {
        match self.get(k) {
            None => Ok(default),
            Some(v) => v.parse().map_err(|_| DeferError::Parse(format!("invalid value for --{k}: {v}"))),
        }
    }
}

fn main() {
    let result = Args::parse().and_then(|args| {
        if args.has("help") || args.cmd == "help" {
            print!("{USAGE}");
            return Ok(());
        }
        match args.cmd.as_str() {
            "run" | "generate" => cmd_run(&args),
            "checkout" => cmd_checkout(&args),
            "verify" => cmd_verify(&args),
            "inspect" => cmd_inspect(&args),
            "explain" => cmd_explain(&args),
            "make-test-model" => cmd_make_test_model(&args),
            other => Err(DeferError::Parse(format!("unknown command `{other}` (see --help)"))),
        }
    });
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------- helpers

fn load_model(args: &Args) -> Result<Model> {
    let path = PathBuf::from(args.req("model")?);
    let t = Instant::now();
    let model = Model::load(&path)?;
    if !args.has("quiet") {
        let c = &model.config;
        eprintln!(
            "[defer] loaded {} ({} layers, d={}, heads={}/{}, vocab={}) in {:.2?}",
            path.display(),
            c.num_layers,
            c.hidden_size,
            c.num_heads,
            c.num_kv_heads,
            c.vocab_size,
            t.elapsed()
        );
        eprintln!("[defer] model signature {}", to_hex(&model.signature));
    }
    Ok(model)
}

fn load_tokenizer(args: &Args, model: &Model) -> Result<Tokenizer> {
    let explicit = args.get("tokenizer").map(PathBuf::from);
    let default = Path::new(args.req("model")?).with_file_name("tokenizer.json");
    let tok = match explicit {
        Some(p) => Tokenizer::load(&p)?,
        None if default.exists() => Tokenizer::load(&default)?,
        None => {
            if model.config.vocab_size < 256 {
                return Err(DeferError::Tokenizer("no tokenizer.json found and vocab < 256".into()));
            }
            if model.config.vocab_size != 256 && !args.has("quiet") {
                eprintln!("[defer] warning: no tokenizer.json found, falling back to raw bytes");
            }
            Tokenizer::bytes()
        }
    };
    if tok.vocab_size() > model.config.vocab_size {
        eprintln!(
            "[defer] warning: tokenizer vocab ({}) larger than model vocab ({})",
            tok.vocab_size(),
            model.config.vocab_size
        );
    }
    Ok(tok)
}

fn seed_arg(args: &Args) -> Result<[u8; 32]> {
    if let Some(h) = args.get("seed-hex") {
        let b = from_hex(h).filter(|b| b.len() == 32).ok_or_else(|| {
            DeferError::Parse("--seed-hex must be 64 hex characters".into())
        })?;
        return Ok(b.try_into().unwrap());
    }
    Ok(seed_from_u64(args.num("seed", 0u64)?))
}

fn encode_prompt(args: &Args, tok: &Tokenizer) -> Result<Vec<u32>> {
    let prompt = args.req("prompt")?;
    let mut ids = Vec::new();
    if !args.has("no-bos") {
        if let Some(b) = tok.bos_token() {
            ids.push(b);
        }
    }
    ids.extend(tok.encode(prompt)?);
    if ids.is_empty() {
        return Err(DeferError::Parse("prompt encodes to zero tokens".into()));
    }
    Ok(ids)
}

fn stop_tokens(args: &Args, model: &Model) -> Vec<u32> {
    if args.has("ignore-eos") {
        Vec::new()
    } else {
        model.config.eos_token_ids.clone()
    }
}

/// Streams token bytes to stdout, holding back incomplete UTF-8 sequences.
struct Streamer {
    pending: Vec<u8>,
}

impl Streamer {
    fn new() -> Self {
        Streamer { pending: Vec::new() }
    }
    fn push(&mut self, bytes: &[u8]) {
        self.pending.extend_from_slice(bytes);
        let valid = match std::str::from_utf8(&self.pending) {
            Ok(_) => self.pending.len(),
            Err(e) if e.error_len().is_some() => self.pending.len(), // invalid: flush lossily
            Err(e) => e.valid_up_to(),
        };
        let out: Vec<u8> = self.pending.drain(..valid).collect();
        let mut so = std::io::stdout().lock();
        let _ = so.write_all(String::from_utf8_lossy(&out).as_bytes());
        let _ = so.flush();
    }
    fn finish(&mut self) {
        let rest = std::mem::take(&mut self.pending);
        let mut so = std::io::stdout().lock();
        let _ = so.write_all(String::from_utf8_lossy(&rest).as_bytes());
        let _ = so.write_all(b"\n");
        let _ = so.flush();
    }
}

fn print_text(tok: &Tokenizer, ids: &[u32], skip_special: bool) {
    print!("{}", tok.decode(ids, skip_special));
    let _ = std::io::stdout().flush();
}

/// Generate with streaming output, optional WAL and final checkpoint.
fn run_generation(
    args: &Args,
    session: &mut Session,
    tok: &Tokenizer,
    stop: &[u32],
    mut wal: Option<WalWriter>,
) -> Result<()> {
    let max_tokens: usize = args.num("max-tokens", 128)?;
    let quiet = args.has("quiet");
    let mut streamer = Streamer::new();
    let t = Instant::now();
    let explain = args.has("explain") || args.has("explain-jsonl");
    let generated = if explain {
        // Explanation lines go to stderr; the text is printed once at the
        // end so the two streams do not interleave on a terminal.
        let top: usize = args.num("explain-top", 3)?;
        let mut jsonl = match args.get("explain-jsonl") {
            Some(p) => Some(std::io::BufWriter::new(std::fs::File::create(p)?)),
            None => None,
        };
        let mut text = Vec::new();
        session.enable_tracing();
        let out = session.generate_explained(max_tokens, stop, top, |s, e| {
            if let Some(w) = wal.as_mut() {
                w.append(e.token, s.rng_word_pos())?;
            }
            if !tok.is_special(e.token) {
                text.extend(tok.token_bytes(e.token));
            }
            if args.has("explain") {
                eprintln!("{}", explanation_line(tok, e));
            }
            if let Some(w) = jsonl.as_mut() {
                writeln!(w, "{}", explanation_json(tok, e))?;
                w.flush()?;
            }
            Ok(())
        })?;
        streamer.push(&text);
        out
    } else {
        session.generate(max_tokens, stop, |s, token| {
            if let Some(w) = wal.as_mut() {
                w.append(token, s.rng_word_pos())?;
            }
            if !tok.is_special(token) {
                streamer.push(&tok.token_bytes(token));
            }
            Ok(())
        })?
    };
    streamer.finish();
    let dt = t.elapsed();
    if !quiet {
        eprintln!(
            "[defer] {} tokens in {:.2?} ({:.2} tok/s), context {} tokens, rng word pos {}",
            generated.len(),
            dt,
            generated.len() as f64 / dt.as_secs_f64().max(1e-9),
            session.history().len(),
            session.rng_word_pos()
        );
        eprintln!("[defer] kv-cache fingerprint {}", to_hex(&session.cache().fingerprint()));
    }
    if let Some(p) = args.get("checkpoint") {
        let ck = session.checkpoint();
        ck.save(Path::new(p))?;
        if !quiet {
            eprintln!("[defer] checkpoint written to {p} ({} bytes)", ck.to_bytes().len());
        }
    }
    Ok(())
}

// ------------------------------------------------------------ attribution

/// Debug-quoted text of one token, e.g. `" Paris"` or `"\n"`.
fn tok_text(tok: &Tokenizer, id: u32) -> String {
    format!("{:?}", String::from_utf8_lossy(&tok.token_bytes(id)))
}

fn json_num(x: f64) -> String {
    if x.is_finite() {
        format!("{x}")
    } else {
        "null".into()
    }
}

fn explanation_line(tok: &Tokenizer, e: &Explanation) -> String {
    let parts: Vec<String> = e
        .top
        .iter()
        .map(|s| format!("{} {} {:.3}", s.position, tok_text(tok, s.token), s.attn_value))
        .collect();
    format!(
        "[explain] #{} {} (p={:.3}) <- {}",
        e.position,
        tok_text(tok, e.token),
        e.logprob.exp(),
        parts.join(" | ")
    )
}

fn position_json(tok: &Tokenizer, s: &PositionScore, ablation: Option<f64>) -> String {
    let mut j = format!(
        "{{\"position\":{},\"token\":{},\"text\":{},\"attention\":{},\"attn_value\":{}",
        s.position,
        s.token,
        json::quote(&String::from_utf8_lossy(&tok.token_bytes(s.token))),
        json_num(s.attention as f64),
        json_num(s.attn_value as f64)
    );
    if let Some(d) = ablation {
        j.push_str(&format!(",\"ablation_delta\":{}", json_num(d)));
    }
    j.push('}');
    j
}

fn explanation_json(tok: &Tokenizer, e: &Explanation) -> String {
    let top: Vec<String> = e.top.iter().map(|s| position_json(tok, s, None)).collect();
    format!(
        "{{\"position\":{},\"token\":{},\"text\":{},\"logprob\":{},\"top\":[{}]}}",
        e.position,
        e.token,
        json::quote(&String::from_utf8_lossy(&tok.token_bytes(e.token))),
        json_num(e.logprob),
        top.join(",")
    )
}

/// `defer explain`: offline attribution for any token of a WAL/checkpoint.
fn cmd_explain(args: &Args) -> Result<()> {
    let from = PathBuf::from(args.req("from")?);
    let ckpt = match read_source(&from)? {
        Source::Checkpoint(c) => c,
        Source::Wal(log) => log.to_checkpoint(),
    };
    let model = load_model(args)?;
    if ckpt.model_signature != model.signature {
        return Err(DeferError::Checkpoint(format!(
            "model signature mismatch: {} was recorded with {}",
            from.display(),
            to_hex(&ckpt.model_signature)
        )));
    }
    let tok = load_tokenizer(args, &model)?;
    let hist = &ckpt.history_tokens;
    let n = hist.len() as i64;
    let step: i64 = args.num("step", n - 1)?;
    let step = if step < 0 { n + step } else { step };
    if step < 1 || step >= n {
        return Err(DeferError::Parse(format!("--step must be in 1..{} (or negative from the end)", n - 1)));
    }
    let step = step as usize;
    let method = args.get("method").unwrap_or("attention");
    let (use_attn, use_abl) = match method {
        "attention" => (true, false),
        "ablation" => (false, true),
        "both" => (true, true),
        m => return Err(DeferError::Parse(format!("unknown --method {m} (attention|ablation|both)"))),
    };
    let top: usize = args.num("top", 10)?;
    let context = &hist[..step];
    let target = hist[step];
    let quiet = args.has("quiet");

    let t = Instant::now();
    let (attn, logits) = attribution::attention_for_context(&model, context)?;
    let logprob = attribution::log_softmax_at(&logits, target as usize);

    let mut ablation: Vec<Option<f64>> = vec![None; context.len()];
    if use_abl {
        let window: usize = args.num("window", context.len())?;
        let last = context.len() - 1; // the query token can not be hidden from itself
        let positions: Vec<usize> = (last.saturating_sub(window)..last).collect();
        if !quiet {
            eprintln!("[defer] ablating {} positions...", positions.len());
        }
        let rep = attribution::ablation_attribution(&model, context, target, &positions)?;
        for s in &rep.scores {
            ablation[s.position] = Some(s.delta);
        }
    }
    if !quiet {
        eprintln!("[defer] attribution computed in {:.2?}", t.elapsed());
    }

    // Rank: ablation delta if available, else attention x value.
    let mut rows: Vec<(PositionScore, Option<f64>)> =
        attn.into_iter().map(|s| { let a = ablation[s.position]; (s, a) }).collect();
    if use_abl {
        if !use_attn {
            rows.retain(|(_, a)| a.is_some());
        }
        attribution::rank_by(&mut rows, |(_, a)| a.unwrap_or(f64::NEG_INFINITY), |(s, _)| s.position);
    } else {
        attribution::rank_by(&mut rows, |(s, _)| s.attn_value as f64, |(s, _)| s.position);
    }
    rows.truncate(top);

    let generated = step >= ckpt.prompt_len as usize;
    if args.has("json") {
        let items: Vec<String> = rows
            .iter()
            .map(|(s, a)| position_json(&tok, s, if use_abl { Some(a.unwrap_or(f64::NAN)) } else { None }))
            .collect();
        println!(
            "{{\"position\":{},\"token\":{},\"text\":{},\"generated\":{},\"logprob\":{},\"method\":{},\"positions\":[{}]}}",
            step,
            target,
            json::quote(&String::from_utf8_lossy(&tok.token_bytes(target))),
            generated,
            json_num(logprob),
            json::quote(method),
            items.join(",")
        );
        return Ok(());
    }

    println!(
        "explaining position {step}: {} ({}, logprob {:.4}, p={:.4})",
        tok_text(&tok, target),
        if generated { "generated" } else { "prompt" },
        logprob,
        logprob.exp()
    );
    // Context with the top positions highlighted.
    let marked: Vec<usize> = rows
        .iter()
        .filter(|(s, a)| if use_abl { a.map_or(false, |d| d > 0.0) } else { s.attn_value > 0.0 })
        .map(|(s, _)| s.position)
        .collect();
    let mut ctx = Vec::new();
    for (i, &id) in context.iter().enumerate() {
        let hl = marked.contains(&i);
        if hl {
            ctx.extend_from_slice("⟦".as_bytes());
        }
        ctx.extend(tok.token_bytes(id));
        if hl {
            ctx.extend_from_slice("⟧".as_bytes());
        }
    }
    println!("context: {}\u{2192} {}\n", String::from_utf8_lossy(&ctx), tok_text(&tok, target));
    print!("{:>4}  {:>5}  {:<20}  {:>10}  {:>9}", "rank", "pos", "token", "attn×value", "attention");
    if use_abl {
        print!("  {:>9}", "Δlogprob");
    }
    println!();
    for (r, (s, a)) in rows.iter().enumerate() {
        let mut text = tok_text(&tok, s.token);
        if text.chars().count() > 20 {
            text = text.chars().take(19).collect::<String>() + "…";
        }
        print!("{:>4}  {:>5}  {:<20}  {:>10.4}  {:>9.4}", r + 1, s.position, text, s.attn_value, s.attention);
        if use_abl {
            match a {
                Some(d) => print!("  {:>+9.4}", d),
                None => print!("  {:>9}", "-"),
            }
        }
        println!();
    }
    if use_abl {
        println!("\nΔlogprob > 0: hiding the position lowers p(target), so it supported this token.");
    }
    Ok(())
}

// --------------------------------------------------------------- commands

fn cmd_run(args: &Args) -> Result<()> {
    let model = load_model(args)?;
    let tok = load_tokenizer(args, &model)?;
    let temperature: f32 = args.num("temperature", 0.7)?;
    let seed = seed_arg(args)?;
    let prompt = encode_prompt(args, &tok)?;
    let stop = stop_tokens(args, &model);

    let mut session = Session::new(&model, seed, temperature)?;
    let t = Instant::now();
    session.prompt(&prompt)?;
    if !args.has("quiet") {
        eprintln!("[defer] prompt: {} tokens evaluated in {:.2?}", prompt.len(), t.elapsed());
    }
    let wal = match args.get("wal") {
        Some(p) => Some(WalWriter::create(
            Path::new(p),
            &WalHeader {
                model_signature: model.signature,
                base_seed: seed,
                temperature,
                prompt_len: prompt.len() as u32,
                start_rng_word_pos: session.rng_word_pos(),
                prefix_tokens: prompt.clone(),
            },
            args.has("durable"),
        )?),
        None => None,
    };
    print!("{}", args.req("prompt")?);
    run_generation(args, &mut session, &tok, &stop, wal)
}

enum Source {
    Checkpoint(DeferCheckpoint),
    Wal(WalLog),
}

fn read_source(path: &Path) -> Result<Source> {
    let bytes = std::fs::read(path)?;
    if bytes.starts_with(b"DEFERWAL") {
        Ok(Source::Wal(WalLog::parse(&bytes)?))
    } else {
        Ok(Source::Checkpoint(DeferCheckpoint::from_bytes(&bytes)?))
    }
}

fn cmd_checkout(args: &Args) -> Result<()> {
    let from = PathBuf::from(args.req("from")?);
    let (ckpt, from_wal) = match read_source(&from)? {
        Source::Checkpoint(c) => (c, false),
        Source::Wal(log) => {
            if log.torn_tail {
                eprintln!("[defer] note: discarded torn trailing WAL record");
            }
            (log.to_checkpoint(), true)
        }
    };
    let model = load_model(args)?;
    let tok = load_tokenizer(args, &model)?;
    let quiet = args.has("quiet");

    let t = Instant::now();
    let mut session = Session::checkout(&model, &ckpt)?;
    if !quiet {
        eprintln!(
            "[defer] checked out {} tokens ({} prompt + {} generated) in {:.2?}, replay verified",
            ckpt.history_tokens.len(),
            ckpt.prompt_len,
            ckpt.generated_tokens().len(),
            t.elapsed()
        );
        eprintln!("[defer] kv-cache fingerprint {}", to_hex(&session.cache().fingerprint()));
    }
    print_text(&tok, session.history(), true);

    // Continue logging: append to the source WAL if it is the same file.
    let wal = match args.get("wal") {
        Some(p) if from_wal && Path::new(p) == from => {
            let (_, w) = WalWriter::resume(Path::new(p), args.has("durable"))?;
            Some(w)
        }
        Some(p) => Some(WalWriter::create(Path::new(p), &WalHeader::from_checkpoint(&ckpt), args.has("durable"))?),
        None => None,
    };
    let max_tokens: usize = args.num("max-tokens", 0)?;
    if max_tokens == 0 && wal.is_none() && !args.has("checkpoint") {
        println!();
        return Ok(());
    }
    let mut args_override = Args { cmd: args.cmd.clone(), flags: args.flags.clone(), positional: vec![] };
    args_override.flags.insert("max-tokens".into(), max_tokens.to_string());
    let stop = stop_tokens(args, &model);
    run_generation(&args_override, &mut session, &tok, &stop, wal)
}

fn cmd_verify(args: &Args) -> Result<()> {
    let model = load_model(args)?;
    let tok = load_tokenizer(args, &model)?;
    let temperature: f32 = args.num("temperature", 0.7)?;
    let seed = seed_arg(args)?;
    let runs: usize = args.num("runs", 10)?;
    let max_tokens: usize = args.num("max-tokens", 32)?;
    let prompt = encode_prompt(args, &tok)?;
    let stop = stop_tokens(args, &model);

    let mut reference: Option<(Vec<u32>, [u8; 32])> = None;
    for run in 0..runs {
        let mut s = Session::new(&model, seed, temperature)?;
        s.prompt(&prompt)?;
        let out = s.generate(max_tokens, &stop, |_, _| Ok(()))?;
        let fp = s.cache().fingerprint();
        match &reference {
            None => {
                println!("reference: {:?}", tok.decode(&out, true));
                reference = Some((out, fp));
            }
            Some((r, rfp)) => {
                if *r != out || *rfp != fp {
                    return Err(DeferError::Divergence(format!("run {run} diverged from run 0")));
                }
            }
        }
    }
    // Checkout must reproduce the same cache bits.
    let (out, fp) = reference.unwrap();
    let mut ck_hist = prompt.clone();
    ck_hist.extend(&out);
    let ck = DeferCheckpoint {
        model_signature: model.signature,
        base_seed: seed,
        temperature,
        prompt_len: prompt.len() as u32,
        rng_word_pos: out.len() as u128 * defer::sampler::words_per_sample(temperature),
        history_tokens: ck_hist,
    };
    let restored = Session::checkout(&model, &ck)?;
    if restored.cache().fingerprint() != fp {
        return Err(DeferError::Divergence("checkout KV cache differs from live cache".into()));
    }
    println!("OK: {runs} runs identical, checkout KV cache bit-identical ({})", to_hex(&fp));
    Ok(())
}

fn cmd_inspect(args: &Args) -> Result<()> {
    let path = args
        .positional
        .first()
        .map(String::as_str)
        .or(args.get("from"))
        .ok_or_else(|| DeferError::Parse("usage: defer inspect <file>".into()))?;
    let print_ck = |c: &DeferCheckpoint| {
        println!("model signature : {}", to_hex(&c.model_signature));
        println!("base seed       : {}", to_hex(&c.base_seed));
        println!("temperature     : {}", c.temperature);
        println!("prompt tokens   : {}", c.prompt_len);
        println!("generated tokens: {}", c.generated_tokens().len());
        println!("rng word pos    : {}", c.rng_word_pos);
        println!("history         : {:?}", c.history_tokens);
    };
    match read_source(Path::new(path))? {
        Source::Checkpoint(c) => {
            println!("type            : checkpoint");
            print_ck(&c);
        }
        Source::Wal(log) => {
            println!("type            : write-ahead log");
            println!("records         : {}{}", log.records.len(), if log.torn_tail { " (torn tail discarded)" } else { "" });
            println!("prefix tokens   : {}", log.header.prefix_tokens.len());
            print_ck(&log.to_checkpoint());
        }
    }
    Ok(())
}

fn cmd_make_test_model(args: &Args) -> Result<()> {
    let out = PathBuf::from(args.req("out")?);
    let seed: u64 = args.num("seed", 0)?;
    let cfg = defer::synthetic::tiny_config();
    defer::synthetic::write_model(&out, &cfg, seed)?;
    println!("wrote synthetic test model to {} (byte tokenizer, vocab 256)", out.display());
    Ok(())
}
