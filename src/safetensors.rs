//! Sequential safetensors reader / writer.
//!
//! The file is consumed strictly front-to-back in a single pass: the
//! 8-byte header length, the JSON header, then each tensor's bytes in
//! ascending offset order. Every byte read is fed into SHA-256 so the
//! model signature costs no additional I/O.

use crate::error::{DeferError, Result};
use crate::json::{self, Value};
use crate::sha256::Sha256;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    BF16,
    F64,
}

impl DType {
    fn parse(s: &str) -> Result<Self> {
        Ok(match s {
            "F32" => DType::F32,
            "F16" => DType::F16,
            "BF16" => DType::BF16,
            "F64" => DType::F64,
            other => {
                return Err(DeferError::Model(format!("unsupported safetensors dtype {other}")))
            }
        })
    }
    fn size(self) -> usize {
        match self {
            DType::F16 | DType::BF16 => 2,
            DType::F32 => 4,
            DType::F64 => 8,
        }
    }
}

/// IEEE 754 binary16 -> binary32, exact.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            // subnormal: normalise
            let mut e: i32 = 0;
            let mut m = mant;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            sign | (((127 - 14 + e) as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn convert(dtype: DType, raw: &[u8]) -> Vec<f32> {
    let mut out = Vec::with_capacity(raw.len() / dtype.size());
    match dtype {
        DType::F32 => {
            for c in raw.chunks_exact(4) {
                out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
            }
        }
        DType::F16 => {
            for c in raw.chunks_exact(2) {
                out.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
            }
        }
        DType::BF16 => {
            for c in raw.chunks_exact(2) {
                out.push(bf16_to_f32(u16::from_le_bytes([c[0], c[1]])));
            }
        }
        DType::F64 => {
            for c in raw.chunks_exact(8) {
                let mut b = [0u8; 8];
                b.copy_from_slice(c);
                out.push(f64::from_le_bytes(b) as f32);
            }
        }
    }
    out
}

struct Entry {
    name: String,
    dtype: DType,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// A fully loaded safetensors file, all tensors converted to f32.
pub struct SafeTensors {
    /// Tensor names in on-disk order.
    pub names: Vec<String>,
    tensors: HashMap<String, Tensor>,
    /// `__metadata__` string map, in header order.
    pub metadata: Vec<(String, String)>,
    /// SHA-256 over every byte of the file.
    pub signature: [u8; 32],
}

impl SafeTensors {
    pub fn get(&self, name: &str) -> Option<&Tensor> {
        self.tensors.get(name)
    }

    pub fn take(&mut self, name: &str) -> Option<Tensor> {
        self.tensors.remove(name)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn meta(&self, key: &str) -> Option<&str> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    pub fn load(path: &Path) -> Result<Self> {
        let f = File::open(path)
            .map_err(|e| DeferError::Model(format!("cannot open {}: {e}", path.display())))?;
        let file_len = f.metadata()?.len();
        let mut r = BufReader::with_capacity(1 << 20, f);
        let mut hasher = Sha256::new();

        let mut len_buf = [0u8; 8];
        r.read_exact(&mut len_buf)?;
        hasher.update(&len_buf);
        let header_len = u64::from_le_bytes(len_buf);
        if header_len > 100 * 1024 * 1024 || 8 + header_len > file_len {
            return Err(DeferError::Model("invalid safetensors header length".into()));
        }
        let mut header = vec![0u8; header_len as usize];
        r.read_exact(&mut header)?;
        hasher.update(&header);
        let header_str = std::str::from_utf8(&header)
            .map_err(|_| DeferError::Model("safetensors header is not UTF-8".into()))?;
        let header = json::parse(header_str)?;
        let obj = header
            .as_object()
            .ok_or_else(|| DeferError::Model("safetensors header is not an object".into()))?;

        let mut metadata = Vec::new();
        let mut entries = Vec::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                if let Some(m) = v.as_object() {
                    for (k, val) in m {
                        if let Some(s) = val.as_str() {
                            metadata.push((k.clone(), s.to_string()));
                        }
                    }
                }
                continue;
            }
            let bad = || DeferError::Model(format!("malformed header entry for {name}"));
            let dtype = DType::parse(v.get("dtype").and_then(Value::as_str).ok_or_else(bad)?)?;
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(Value::as_array)
                .ok_or_else(bad)?
                .iter()
                .map(|d| d.as_u64().map(|x| x as usize).ok_or_else(bad))
                .collect::<Result<_>>()?;
            let offs = v.get("data_offsets").and_then(Value::as_array).ok_or_else(bad)?;
            if offs.len() != 2 {
                return Err(bad());
            }
            let start = offs[0].as_u64().ok_or_else(bad)? as usize;
            let end = offs[1].as_u64().ok_or_else(bad)? as usize;
            let numel: usize = shape.iter().product();
            if end < start || end - start != numel * dtype.size() {
                return Err(DeferError::Model(format!("size mismatch for tensor {name}")));
            }
            entries.push(Entry { name: name.clone(), dtype, shape, start, end });
        }
        // Sequential on-disk order; ties broken by name for total ordering.
        entries.sort_by(|a, b| a.start.cmp(&b.start).then_with(|| a.name.cmp(&b.name)));

        let data_len = (file_len - 8 - header_len) as usize;
        let mut pos = 0usize;
        let mut names = Vec::with_capacity(entries.len());
        let mut tensors = HashMap::with_capacity(entries.len());
        let mut buf = Vec::new();
        for e in entries {
            if e.start < pos || e.end > data_len {
                return Err(DeferError::Model(format!("overlapping/out of range tensor {}", e.name)));
            }
            if e.start > pos {
                skip(&mut r, &mut hasher, e.start - pos)?;
            }
            buf.resize(e.end - e.start, 0);
            r.read_exact(&mut buf)?;
            hasher.update(&buf);
            pos = e.end;
            let data = convert(e.dtype, &buf);
            let t = if e.shape.is_empty() {
                Tensor::from_1d(data)
            } else if e.shape.len() > 3 {
                // Flatten leading dims: the engine only needs rank <= 3.
                let last = *e.shape.last().unwrap();
                let n = data.len();
                Tensor::new(data, &[n / last.max(1), last])?
            } else {
                Tensor::new(data, &e.shape)?
            };
            names.push(e.name.clone());
            tensors.insert(e.name, t);
        }
        if pos < data_len {
            skip(&mut r, &mut hasher, data_len - pos)?;
        }
        Ok(SafeTensors { names, tensors, metadata, signature: hasher.finalize() })
    }
}

fn skip<R: Read>(r: &mut R, hasher: &mut Sha256, mut n: usize) -> Result<()> {
    let mut chunk = vec![0u8; 64 * 1024];
    while n > 0 {
        let k = n.min(chunk.len());
        r.read_exact(&mut chunk[..k])?;
        hasher.update(&chunk[..k]);
        n -= k;
    }
    Ok(())
}

/// Write f32 tensors to a safetensors file (in the given order).
pub fn write(
    path: &Path,
    tensors: &[(String, Vec<usize>, Vec<f32>)],
    metadata: &[(String, String)],
) -> Result<()> {
    let mut header = String::from("{");
    if !metadata.is_empty() {
        header.push_str("\"__metadata__\":{");
        for (i, (k, v)) in metadata.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            header.push_str(&format!("{}:{}", json::quote(k), json::quote(v)));
        }
        header.push('}');
    }
    let mut offset = 0usize;
    for (i, (name, shape, data)) in tensors.iter().enumerate() {
        if i > 0 || !metadata.is_empty() {
            header.push(',');
        }
        let n: usize = shape.iter().product();
        assert_eq!(n, data.len(), "tensor {name}: shape/data mismatch");
        let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        header.push_str(&format!(
            "{}:{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            json::quote(name),
            dims.join(","),
            offset,
            offset + n * 4
        ));
        offset += n * 4;
    }
    header.push('}');
    while header.len() % 8 != 0 {
        header.push(' ');
    }
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(&(header.len() as u64).to_le_bytes())?;
    w.write_all(header.as_bytes())?;
    for (_, _, data) in tensors {
        for v in data {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_conversions() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x7BFF), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
    }

    #[test]
    fn write_then_read() {
        let dir = std::env::temp_dir().join(format!("defer-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.safetensors");
        let tensors = vec![
            ("b".to_string(), vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
            ("a".to_string(), vec![3], vec![-1.0, 0.5, 9.0]),
        ];
        write(&p, &tensors, &[("k".into(), "v".into())]).unwrap();
        let st = SafeTensors::load(&p).unwrap();
        assert_eq!(st.names, vec!["b", "a"]);
        assert_eq!(st.get("b").unwrap().shape(), &[2, 2]);
        assert_eq!(st.get("a").unwrap().data(), &[-1.0, 0.5, 9.0]);
        assert_eq!(st.meta("k"), Some("v"));
        let bytes = std::fs::read(&p).unwrap();
        assert_eq!(st.signature, Sha256::digest(&bytes));
        std::fs::remove_dir_all(&dir).ok();
    }
}
