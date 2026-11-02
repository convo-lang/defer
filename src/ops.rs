//! Naive, strictly sequential math kernels.
//!
//! Every reduction is a single accumulator walked in index order. Rust does
//! not permit LLVM to reassociate floating point additions (there is no
//! fast-math), and does not contract `a * b + c` into FMA, so the evaluation
//! order written here is exactly the order executed on every run.

/// `out[m,n] = a[m,k] * b[k,n]`, all row-major.
pub fn matmul_sequential(out: &mut [f32], a: &[f32], b: &[f32], m: usize, n: usize, k: usize) {
    assert_eq!(out.len(), m * n);
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }
}

/// Linear layer in PyTorch layout: `w` is `[out_dim, in_dim]`, computes
/// `out[o] = sum_i w[o,i] * x[i] (+ bias[o])`.
pub fn linear(out: &mut [f32], x: &[f32], w: &[f32], bias: Option<&[f32]>) {
    let in_dim = x.len();
    let out_dim = out.len();
    assert_eq!(w.len(), in_dim * out_dim, "linear: weight shape mismatch");
    // Four *independent* output rows are computed per pass so the CPU can
    // overlap their dependency chains. Each row still has exactly one
    // accumulator summed in index order 0..in_dim, so every output value
    // is bit-identical to the plain one-row-at-a-time loop below.
    let blocks = out_dim / 4;
    for blk in 0..blocks {
        let o = blk * 4;
        let r0 = &w[o * in_dim..(o + 1) * in_dim];
        let r1 = &w[(o + 1) * in_dim..(o + 2) * in_dim];
        let r2 = &w[(o + 2) * in_dim..(o + 3) * in_dim];
        let r3 = &w[(o + 3) * in_dim..(o + 4) * in_dim];
        let (mut s0, mut s1, mut s2, mut s3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        for i in 0..in_dim {
            let xi = x[i];
            s0 += r0[i] * xi;
            s1 += r1[i] * xi;
            s2 += r2[i] * xi;
            s3 += r3[i] * xi;
        }
        if let Some(b) = bias {
            s0 += b[o];
            s1 += b[o + 1];
            s2 += b[o + 2];
            s3 += b[o + 3];
        }
        out[o] = s0;
        out[o + 1] = s1;
        out[o + 2] = s2;
        out[o + 3] = s3;
    }
    for o in blocks * 4..out_dim {
        let row = &w[o * in_dim..(o + 1) * in_dim];
        let mut sum = 0.0f32;
        for i in 0..in_dim {
            sum += row[i] * x[i];
        }
        if let Some(b) = bias {
            sum += b[o];
        }
        out[o] = sum;
    }
}

/// Sequential dot product.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// RMSNorm: `out = x / sqrt(mean(x^2) + eps) * w`.
pub fn rmsnorm(out: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    assert_eq!(x.len(), w.len());
    assert_eq!(out.len(), x.len());
    let mut ss = 0.0f32;
    for i in 0..x.len() {
        ss += x[i] * x[i];
    }
    let inv = 1.0 / (ss / x.len() as f32 + eps).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * w[i];
    }
}

/// LayerNorm: `out = (x - mean) / sqrt(var + eps) * w + b`.
pub fn layernorm(out: &mut [f32], x: &[f32], w: &[f32], b: &[f32], eps: f32) {
    assert_eq!(x.len(), w.len());
    assert_eq!(x.len(), b.len());
    assert_eq!(out.len(), x.len());
    let n = x.len() as f32;
    let mut mean = 0.0f32;
    for &v in x {
        mean += v;
    }
    mean /= n;
    let mut var = 0.0f32;
    for &v in x {
        let d = v - mean;
        var += d * d;
    }
    var /= n;
    let inv = 1.0 / (var + eps).sqrt();
    for i in 0..x.len() {
        out[i] = (x[i] - mean) * inv * w[i] + b[i];
    }
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// GeLU, tanh approximation (as used by GPT-2 style models).
pub fn gelu(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

pub fn silu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = silu(*v);
    }
}

pub fn gelu_inplace(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v = gelu(*v);
    }
}

/// `a[i] += b[i]`
pub fn add_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] += b[i];
    }
}

/// `a[i] *= b[i]`
pub fn mul_inplace(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for i in 0..a.len() {
        a[i] *= b[i];
    }
}

/// Numerically stable, sequential in-place softmax.
pub fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }
    let mut max = f32::NEG_INFINITY;
    for &v in x.iter() {
        if v > max {
            max = v;
        }
    }
    let mut sum = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in x.iter_mut() {
        *v *= inv;
    }
}

/// Bit-level slice equality.
pub fn bit_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_small() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
        let mut out = [0.0; 4];
        matmul_sequential(&mut out, &a, &b, 2, 2, 3);
        assert_eq!(out, [58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn linear_matches_matmul() {
        // w: [2 out, 3 in]
        let w = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 0.5, -1.0];
        let mut out = [0.0; 2];
        linear(&mut out, &x, &w, Some(&[0.5, -0.5]));
        assert_eq!(out, [1.0 + 1.0 - 3.0 + 0.5, 4.0 + 2.5 - 6.0 - 0.5]);
    }

    #[test]
    fn blocked_linear_is_bit_identical_to_reference() {
        let (in_dim, out_dim) = (37, 11);
        let w: Vec<f32> = (0..in_dim * out_dim).map(|i| ((i * 7919) % 1000) as f32 * 1e-3 - 0.5).collect();
        let x: Vec<f32> = (0..in_dim).map(|i| ((i * 104729) % 997) as f32 * 1e-2 - 4.0).collect();
        let b: Vec<f32> = (0..out_dim).map(|i| i as f32 * 0.1).collect();
        let mut out = vec![0.0; out_dim];
        linear(&mut out, &x, &w, Some(&b));
        for o in 0..out_dim {
            let mut s = 0.0f32;
            for i in 0..in_dim {
                s += w[o * in_dim + i] * x[i];
            }
            s += b[o];
            assert_eq!(out[o].to_bits(), s.to_bits());
        }
    }

    #[test]
    fn norms() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let mut out = [0.0; 4];
        rmsnorm(&mut out, &x, &[1.0; 4], 0.0);
        let rms = (30.0f32 / 4.0).sqrt();
        for i in 0..4 {
            assert!((out[i] - x[i] / rms).abs() < 1e-6);
        }
        layernorm(&mut out, &x, &[1.0; 4], &[0.0; 4], 0.0);
        let s: f32 = out.iter().sum();
        assert!(s.abs() < 1e-5);
    }

    #[test]
    fn activations_and_softmax() {
        assert_eq!(silu(0.0), 0.0);
        assert!((gelu(1.0) - 0.841_192).abs() < 1e-4);
        let mut v = [1.0, 2.0, 3.0];
        softmax_inplace(&mut v);
        assert!((v.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(v[2] > v[1] && v[1] > v[0]);
        let mut a = [1.0, 2.0];
        add_inplace(&mut a, &[1.0, 1.0]);
        mul_inplace(&mut a, &[2.0, 3.0]);
        assert_eq!(a, [4.0, 9.0]);
    }
}
