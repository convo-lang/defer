//! Byte-pair-encoding tokenizer loaded from a HuggingFace `tokenizer.json`.
//!
//! Supports two BPE flavours plus a trivial fallback:
//! * **Byte-level BPE** (GPT-2 / Llama-3 / Qwen): bytes are mapped to
//!   printable code points, text is pre-split with a GPT-2 style splitter.
//! * **SentencePiece BPE** (Llama-2 / Mistral / TinyLlama): spaces become
//!   `▁`, unknown pieces fall back to `<0xXX>` byte tokens.
//! * **Raw bytes**: token id == byte value (used by the synthetic model).
//!
//! Merging always applies the lowest-ranked pair, leftmost first, so
//! encoding is a pure function of (vocab, merges, text).

use crate::error::{DeferError, Result};
use crate::json::{self, Value};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Bytes,
    ByteLevel,
    SentencePiece,
}

pub struct Tokenizer {
    mode: Mode,
    vocab: HashMap<String, u32>,
    id_to_token: Vec<Option<String>>,
    merges: HashMap<(String, String), u32>,
    /// Added/special tokens, sorted longest-first then by id for matching.
    added: Vec<(String, u32)>,
    special_ids: Vec<u32>,
    byte_fallback: [Option<u32>; 256],
    unk: Option<u32>,
    add_prefix_space: bool,
    template_bos: Option<u32>,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
}

/// GPT-2's reversible byte -> unicode mapping.
fn bytes_to_unicode() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut n = 0u32;
    for b in 0..256u32 {
        let printable = (b'!' as u32..=b'~' as u32).contains(&b)
            || (0xA1..=0xAC).contains(&b)
            || (0xAE..=0xFF).contains(&b);
        table[b as usize] = if printable {
            char::from_u32(b).unwrap()
        } else {
            let c = char::from_u32(256 + n).unwrap();
            n += 1;
            c
        };
    }
    table
}

fn contains_type(v: &Value, ty: &str) -> bool {
    match v {
        Value::Object(pairs) => pairs.iter().any(|(k, val)| {
            (k == "type" && val.as_str() == Some(ty)) || contains_type(val, ty)
        }),
        Value::Array(items) => items.iter().any(|x| contains_type(x, ty)),
        _ => false,
    }
}

fn find_key<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    match v {
        Value::Object(pairs) => pairs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
            .or_else(|| pairs.iter().find_map(|(_, v)| find_key(v, key))),
        Value::Array(items) => items.iter().find_map(|x| find_key(x, key)),
        _ => None,
    }
}

impl Tokenizer {
    /// Identity byte tokenizer: ids 0..=255 are raw bytes.
    pub fn bytes() -> Self {
        let enc = bytes_to_unicode();
        Tokenizer {
            mode: Mode::Bytes,
            vocab: HashMap::new(),
            id_to_token: Vec::new(),
            merges: HashMap::new(),
            added: Vec::new(),
            special_ids: Vec::new(),
            byte_fallback: [None; 256],
            unk: None,
            add_prefix_space: false,
            template_bos: None,
            byte_encoder: enc,
            byte_decoder: HashMap::new(),
        }
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn vocab_size(&self) -> usize {
        match self.mode {
            Mode::Bytes => 256,
            _ => self.id_to_token.len(),
        }
    }

    /// BOS token the tokenizer's post-processor would prepend, if any.
    pub fn bos_token(&self) -> Option<u32> {
        self.template_bos
    }

    pub fn is_special(&self, id: u32) -> bool {
        self.special_ids.contains(&id)
    }

    pub fn load(path: &Path) -> Result<Self> {
        let src = std::fs::read_to_string(path)
            .map_err(|e| DeferError::Tokenizer(format!("cannot read {}: {e}", path.display())))?;
        Self::from_json_str(&src)
    }

    pub fn from_json_str(src: &str) -> Result<Self> {
        let root = json::parse(src)?;
        let err = |m: &str| DeferError::Tokenizer(m.to_string());
        let model = root.get("model").ok_or_else(|| err("missing `model`"))?;
        if let Some(t) = model.get("type").and_then(Value::as_str) {
            if t != "BPE" {
                return Err(err(&format!("unsupported tokenizer model type {t} (only BPE)")));
            }
        }
        let vocab_obj = model
            .get("vocab")
            .and_then(Value::as_object)
            .ok_or_else(|| err("missing model.vocab"))?;
        let mut vocab = HashMap::with_capacity(vocab_obj.len());
        let mut id_to_token: Vec<Option<String>> = Vec::new();
        let mut put = |tok: &str, id: u32, vocab: &mut HashMap<String, u32>| {
            let i = id as usize;
            if id_to_token.len() <= i {
                id_to_token.resize(i + 1, None);
            }
            id_to_token[i] = Some(tok.to_string());
            vocab.insert(tok.to_string(), id);
        };
        for (tok, id) in vocab_obj {
            let id = id.as_u64().ok_or_else(|| err("non-integer vocab id"))? as u32;
            put(tok, id, &mut vocab);
        }

        let mut merges = HashMap::new();
        if let Some(list) = model.get("merges").and_then(Value::as_array) {
            for (rank, m) in list.iter().enumerate() {
                let pair = match m {
                    Value::String(s) => {
                        let mut it = s.splitn(2, ' ');
                        match (it.next(), it.next()) {
                            (Some(a), Some(b)) => (a.to_string(), b.to_string()),
                            _ => return Err(err("malformed merge")),
                        }
                    }
                    Value::Array(a) if a.len() == 2 => (
                        a[0].as_str().ok_or_else(|| err("malformed merge"))?.to_string(),
                        a[1].as_str().ok_or_else(|| err("malformed merge"))?.to_string(),
                    ),
                    _ => return Err(err("malformed merge")),
                };
                // Keep the first (lowest) rank if a pair is duplicated.
                merges.entry(pair).or_insert(rank as u32);
            }
        }

        let mut added = Vec::new();
        let mut special_ids = Vec::new();
        if let Some(list) = root.get("added_tokens").and_then(Value::as_array) {
            for t in list {
                let (Some(content), Some(id)) =
                    (t.get("content").and_then(Value::as_str), t.get("id").and_then(Value::as_u64))
                else {
                    continue;
                };
                put(content, id as u32, &mut vocab);
                if !content.is_empty() {
                    added.push((content.to_string(), id as u32));
                }
                if t.get("special").and_then(Value::as_bool).unwrap_or(false) {
                    special_ids.push(id as u32);
                }
            }
        }
        added.sort_by(|a, b| b.0.len().cmp(&a.0.len()).then(a.1.cmp(&b.1)));

        let byte_level = root.get("pre_tokenizer").map_or(false, |v| contains_type(v, "ByteLevel"))
            || root.get("decoder").map_or(false, |v| contains_type(v, "ByteLevel"));
        let mode = if byte_level { Mode::ByteLevel } else { Mode::SentencePiece };

        let mut byte_fallback = [None; 256];
        if model.get("byte_fallback").and_then(Value::as_bool).unwrap_or(mode == Mode::SentencePiece) {
            for b in 0..256usize {
                byte_fallback[b] = vocab.get(&format!("<0x{b:02X}>")).copied();
            }
        }
        let unk = model.get("unk_token").and_then(Value::as_str).and_then(|u| vocab.get(u).copied());

        // SentencePiece-style tokenizers prepend "▁" unless told otherwise.
        let add_prefix_space = mode == Mode::SentencePiece && {
            let explicit_off = root.get("pre_tokenizer").map_or(false, |p| {
                p.get("prepend_scheme").and_then(Value::as_str) == Some("never")
                    || p.get("add_prefix_space").and_then(Value::as_bool) == Some(false)
            });
            let normalizer_prepends = root
                .get("normalizer")
                .map_or(false, |n| contains_type(n, "Prepend"));
            let metaspace = root.get("pre_tokenizer").map_or(false, |p| contains_type(p, "Metaspace"));
            !explicit_off && (normalizer_prepends || metaspace || root.get("normalizer").map_or(true, Value::is_null))
        };

        // BOS from a TemplateProcessing post-processor whose `single`
        // template starts with a special token (Llama-2, Llama-3, ...).
        let template_bos = root
            .get("post_processor")
            .and_then(|p| find_key(p, "single"))
            .and_then(Value::as_array)
            .and_then(|a| a.first())
            .and_then(|f| f.get("SpecialToken"))
            .and_then(|st| st.get("id"))
            .and_then(Value::as_str)
            .and_then(|tok| vocab.get(tok).copied());

        let enc = bytes_to_unicode();
        let mut byte_decoder = HashMap::with_capacity(256);
        for (b, &c) in enc.iter().enumerate() {
            byte_decoder.insert(c, b as u8);
        }
        Ok(Tokenizer {
            mode,
            vocab,
            id_to_token,
            merges,
            added,
            special_ids,
            byte_fallback,
            unk,
            add_prefix_space,
            template_bos,
            byte_encoder: enc,
            byte_decoder,
        })
    }

    /// Encode text to token ids (no BOS added).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        if self.mode == Mode::Bytes {
            return Ok(text.bytes().map(|b| b as u32).collect());
        }
        let mut out = Vec::new();
        let mut rest = text;
        let mut first = true;
        while !rest.is_empty() {
            // Find the earliest added-token occurrence (longest on ties).
            let mut hit: Option<(usize, &str, u32)> = None;
            for (s, id) in &self.added {
                if let Some(p) = rest.find(s.as_str()) {
                    if hit.map_or(true, |(hp, _, _)| p < hp) {
                        hit = Some((p, s, *id));
                    }
                }
            }
            let (seg, special) = match hit {
                Some((p, s, id)) => (&rest[..p], Some((s.len(), id, p))),
                None => (rest, None),
            };
            if !seg.is_empty() {
                self.encode_segment(seg, first, &mut out)?;
            }
            first = false;
            match special {
                Some((len, id, p)) => {
                    out.push(id);
                    rest = &rest[p + len..];
                }
                None => break,
            }
        }
        Ok(out)
    }

    fn encode_segment(&self, seg: &str, first: bool, out: &mut Vec<u32>) -> Result<()> {
        match self.mode {
            Mode::Bytes => unreachable!(),
            Mode::ByteLevel => {
                for piece in pretokenize(seg) {
                    let mapped: String = piece.bytes().map(|b| self.byte_encoder[b as usize]).collect();
                    for sym in self.bpe(&mapped) {
                        self.push_symbol(&sym, out)?;
                    }
                }
            }
            Mode::SentencePiece => {
                let mut s = String::with_capacity(seg.len() + 3);
                if first && self.add_prefix_space {
                    s.push('▁');
                }
                for c in seg.chars() {
                    s.push(if c == ' ' { '▁' } else { c });
                }
                for sym in self.bpe(&s) {
                    self.push_symbol(&sym, out)?;
                }
            }
        }
        Ok(())
    }

    fn push_symbol(&self, sym: &str, out: &mut Vec<u32>) -> Result<()> {
        if let Some(&id) = self.vocab.get(sym) {
            out.push(id);
            return Ok(());
        }
        // Byte fallback, then per-char lookup, then <unk>.
        let bytes = sym.as_bytes();
        if bytes.iter().all(|&b| self.byte_fallback[b as usize].is_some()) {
            for &b in bytes {
                out.push(self.byte_fallback[b as usize].unwrap());
            }
            return Ok(());
        }
        for c in sym.chars() {
            let mut buf = [0u8; 4];
            match self.vocab.get(c.encode_utf8(&mut buf) as &str) {
                Some(&id) => out.push(id),
                None => match self.unk {
                    Some(u) => out.push(u),
                    None => {
                        return Err(DeferError::Tokenizer(format!("cannot encode symbol {c:?}")))
                    }
                },
            }
        }
        Ok(())
    }

    /// Classic BPE: repeatedly merge the lowest-rank adjacent pair (leftmost
    /// on equal rank) until no mergeable pair remains.
    fn bpe(&self, word: &str) -> Vec<String> {
        let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
        if syms.len() < 2 {
            return syms;
        }
        let mut key = (String::new(), String::new());
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..syms.len() - 1 {
                key.0.clear();
                key.0.push_str(&syms[i]);
                key.1.clear();
                key.1.push_str(&syms[i + 1]);
                if let Some(&rank) = self.merges.get(&key) {
                    if best.map_or(true, |(r, _)| rank < r) {
                        best = Some((rank, i));
                    }
                }
            }
            match best {
                None => break,
                Some((_, i)) => {
                    let right = syms.remove(i + 1);
                    syms[i].push_str(&right);
                    if syms.len() < 2 {
                        break;
                    }
                }
            }
        }
        syms
    }

    /// Raw bytes a single token contributes to the output stream.
    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        match self.mode {
            Mode::Bytes => {
                if id < 256 {
                    vec![id as u8]
                } else {
                    Vec::new()
                }
            }
            _ => {
                let Some(Some(tok)) = self.id_to_token.get(id as usize) else {
                    return Vec::new();
                };
                if self.added.iter().any(|(_, a)| *a == id) {
                    return tok.as_bytes().to_vec();
                }
                match self.mode {
                    Mode::ByteLevel => tok
                        .chars()
                        .map(|c| self.byte_decoder.get(&c).copied())
                        .collect::<Option<Vec<u8>>>()
                        .unwrap_or_else(|| tok.as_bytes().to_vec()),
                    _ => {
                        let b = tok.as_bytes();
                        if b.len() == 6 && tok.starts_with("<0x") && tok.ends_with('>') {
                            if let Ok(v) = u8::from_str_radix(&tok[3..5], 16) {
                                return vec![v];
                            }
                        }
                        tok.replace('▁', " ").into_bytes()
                    }
                }
            }
        }
    }

    /// Decode a full token sequence. Special tokens are skipped if
    /// `skip_special` is set.
    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if skip_special && self.is_special(id) {
                continue;
            }
            bytes.extend(self.token_bytes(id));
        }
        if self.mode == Mode::SentencePiece && self.add_prefix_space && bytes.first() == Some(&b' ') {
            bytes.remove(0);
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Approximation of the GPT-2 pre-tokenizer regex:
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
pub fn pretokenize(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let byte_at = |i: usize| if i < n { chars[i].0 } else { text.len() };
    let mut out = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i].1;
        // Contractions.
        if c == '\'' {
            let rest = &text[chars[i].0..];
            let lower: String = rest.chars().take(3).collect::<String>().to_lowercase();
            let len = ["'re", "'ve", "'ll"]
                .iter()
                .find(|p| lower.starts_with(*p))
                .map(|_| 3)
                .or_else(|| ["'s", "'t", "'m", "'d"].iter().find(|p| lower.starts_with(*p)).map(|_| 2));
            if let Some(l) = len {
                out.push(&text[chars[i].0..byte_at(i + l)]);
                i += l;
                continue;
            }
        }
        // Optional leading space followed by a non-space run.
        let mut j = i;
        if c == ' ' && i + 1 < n && !chars[i + 1].1.is_whitespace() {
            j = i + 1;
        }
        let d = chars[j].1;
        if !d.is_whitespace() {
            let class = |x: char| -> u8 {
                if x.is_alphabetic() {
                    0
                } else if x.is_numeric() {
                    1
                } else if x.is_whitespace() {
                    3
                } else {
                    2
                }
            };
            let k0 = class(d);
            let mut k = j + 1;
            while k < n && class(chars[k].1) == k0 {
                k += 1;
            }
            out.push(&text[chars[i].0..byte_at(k)]);
            i = k;
            continue;
        }
        // Whitespace run; leave the final space for the next word.
        let mut k = i;
        while k < n && chars[k].1.is_whitespace() {
            k += 1;
        }
        if k < n && k - i > 1 {
            k -= 1;
        }
        out.push(&text[chars[i].0..byte_at(k)]);
        i = k;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretokenizer() {
        assert_eq!(
            pretokenize("Hello world, it's 2024!  ok\n"),
            vec!["Hello", " world", ",", " it", "'s", " 2024", "!", " ", " ok", "\n"]
        );
    }

    fn byte_level_json() -> String {
        // vocab: all single byte-chars for "a", "b", "Ġ" plus merges.
        r#"{
          "added_tokens": [{"id": 6, "content": "<|end|>", "special": true}],
          "pre_tokenizer": {"type": "ByteLevel"},
          "decoder": {"type": "ByteLevel"},
          "model": {"type": "BPE",
            "vocab": {"a": 0, "b": 1, "Ġ": 2, "ab": 3, "Ġab": 4, "Ġa": 5},
            "merges": ["a b", "Ġ a", "Ġ ab"]}
        }"#
        .to_string()
    }

    #[test]
    fn byte_level_bpe() {
        let t = Tokenizer::from_json_str(&byte_level_json()).unwrap();
        assert_eq!(t.mode(), Mode::ByteLevel);
        let ids = t.encode("ab ab a<|end|>b").unwrap();
        // "a b" (rank 0) merges before "Ġ a" (rank 1), so " ab" -> Ġ + ab -> Ġab.
        assert_eq!(ids, vec![3, 4, 5, 6, 1]);
        assert_eq!(t.decode(&ids, false), "ab ab a<|end|>b");
        assert_eq!(t.decode(&ids, true), "ab ab ab");
    }

    #[test]
    fn sentencepiece_bpe_with_byte_fallback() {
        let src = r#"{
          "normalizer": {"type": "Sequence", "normalizers": [{"type": "Prepend", "prepend": "▁"}, {"type": "Replace"}]},
          "model": {"type": "BPE", "byte_fallback": true, "unk_token": "<unk>",
            "vocab": {"<unk>": 0, "▁": 1, "h": 2, "i": 3, "▁h": 4, "▁hi": 5, "<0x21>": 6, "<0xC3>": 7, "<0xA9>": 8},
            "merges": [["▁", "h"], ["▁h", "i"]]}
        }"#;
        let t = Tokenizer::from_json_str(src).unwrap();
        assert_eq!(t.mode(), Mode::SentencePiece);
        let ids = t.encode("hi hi!é").unwrap();
        assert_eq!(ids, vec![5, 5, 6, 7, 8]);
        assert_eq!(t.decode(&ids, false), "hi hi!é");
    }

    #[test]
    fn bytes_roundtrip() {
        let t = Tokenizer::bytes();
        let ids = t.encode("héllo").unwrap();
        assert_eq!(t.decode(&ids, false), "héllo");
    }
}
