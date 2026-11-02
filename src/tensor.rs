//! Minimal tensor abstraction: a contiguous, row-major `Vec<f32>` plus a
//! shape of rank 1, 2 or 3. The flat layout is the only layout, so there is
//! never any stride/permutation divergence between code paths.

use crate::error::{DeferError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

fn check_rank(shape: &[usize]) -> Result<()> {
    if shape.is_empty() || shape.len() > 3 {
        return Err(DeferError::Model(format!(
            "tensor rank must be 1, 2 or 3 (got shape {shape:?})"
        )));
    }
    Ok(())
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: &[usize]) -> Result<Self> {
        check_rank(shape)?;
        let n: usize = shape.iter().product();
        if n != data.len() {
            return Err(DeferError::Model(format!(
                "shape {shape:?} needs {n} elements but data has {}",
                data.len()
            )));
        }
        Ok(Tensor { data, shape: shape.to_vec() })
    }

    pub fn zeros(shape: &[usize]) -> Result<Self> {
        let n = shape.iter().product();
        Tensor::new(vec![0.0; n], shape)
    }

    pub fn from_1d(data: Vec<f32>) -> Self {
        let n = data.len();
        Tensor { data, shape: vec![n] }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// Number of rows when viewed as a 2D matrix `[rows, last_dim]`.
    pub fn rows(&self) -> usize {
        self.data.len() / self.cols().max(1)
    }

    /// Size of the innermost dimension.
    pub fn cols(&self) -> usize {
        *self.shape.last().unwrap_or(&0)
    }

    /// Borrow row `i` of the tensor viewed as `[rows, cols]`.
    pub fn row(&self, i: usize) -> &[f32] {
        let c = self.cols();
        &self.data[i * c..(i + 1) * c]
    }

    pub fn at2(&self, i: usize, j: usize) -> f32 {
        debug_assert_eq!(self.rank(), 2);
        self.data[i * self.shape[1] + j]
    }

    pub fn at3(&self, i: usize, j: usize, k: usize) -> f32 {
        debug_assert_eq!(self.rank(), 3);
        self.data[(i * self.shape[1] + j) * self.shape[2] + k]
    }

    pub fn reshape(mut self, shape: &[usize]) -> Result<Self> {
        check_rank(shape)?;
        let n: usize = shape.iter().product();
        if n != self.data.len() {
            return Err(DeferError::Model(format!(
                "cannot reshape {:?} into {shape:?}",
                self.shape
            )));
        }
        self.shape = shape.to_vec();
        Ok(self)
    }

    /// Bit-level equality (distinguishes -0.0 / 0.0 and NaN payloads).
    pub fn bit_eq(&self, other: &Tensor) -> bool {
        self.shape == other.shape && crate::ops::bit_eq(&self.data, &other.data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes() {
        let t = Tensor::new((0..24).map(|x| x as f32).collect(), &[2, 3, 4]).unwrap();
        assert_eq!(t.at3(1, 2, 3), 23.0);
        assert_eq!(t.rows(), 6);
        let t = t.reshape(&[6, 4]).unwrap();
        assert_eq!(t.at2(5, 3), 23.0);
        assert_eq!(t.row(1), &[4.0, 5.0, 6.0, 7.0]);
        assert!(Tensor::zeros(&[1, 2, 3, 4]).is_err());
        assert!(Tensor::new(vec![0.0; 3], &[2, 2]).is_err());
    }
}
