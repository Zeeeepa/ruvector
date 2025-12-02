//! FastGRNN Router for intelligent resource allocation
//!
//! Implements a FastGRNN (Fast, Accurate, Stable, and Tiny GRU) based router
//! that learns to select optimal model size, context size, and generation
//! parameters based on query characteristics.

use crate::config::RouterConfig;
use crate::error::{Error, Result, RouterError};
use crate::types::{ModelSize, RoutingDecision, CONTEXT_BINS};

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// FastGRNN Router for dynamic resource allocation
#[derive(Debug)]
pub struct FastGRNNRouter {
    /// Cell parameters
    cell: FastGRNNCell,
    /// Output heads
    output_heads: OutputHeads,
    /// Input normalization parameters
    input_norm: LayerNorm,
    /// Configuration
    config: RouterConfig,
}

/// FastGRNN cell implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct FastGRNNCell {
    /// Input-to-update gate weights (sparse)
    w_z: SparseMatrix,
    /// Recurrent-to-update gate weights (low-rank)
    u_z: LowRankMatrix,
    /// Update gate bias
    b_z: Vec<f32>,
    /// Input-to-hidden weights (sparse)
    w_h: SparseMatrix,
    /// Recurrent-to-hidden weights (low-rank)
    u_h: LowRankMatrix,
    /// Hidden bias
    b_h: Vec<f32>,
    /// FastGRNN zeta scalar
    zeta: f32,
    /// FastGRNN nu scalar
    nu: f32,
}

/// Output heads for routing decisions
#[derive(Debug, Clone, Serialize, Deserialize)]
struct OutputHeads {
    /// Model selection: hidden_dim -> 4
    w_model: Vec<f32>,
    b_model: Vec<f32>,
    /// Context selection: hidden_dim -> 5
    w_context: Vec<f32>,
    b_context: Vec<f32>,
    /// Temperature: hidden_dim -> 1
    w_temp: Vec<f32>,
    b_temp: f32,
    /// Top-p: hidden_dim -> 1
    w_top_p: Vec<f32>,
    b_top_p: f32,
    /// Confidence: hidden_dim -> 1
    w_conf: Vec<f32>,
    b_conf: f32,
}

/// Sparse matrix representation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SparseMatrix {
    /// Non-zero indices (row, col)
    indices: Vec<(usize, usize)>,
    /// Non-zero values
    values: Vec<f32>,
    /// Matrix shape (rows, cols)
    shape: (usize, usize),
}

/// Low-rank matrix (U = A @ B)
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LowRankMatrix {
    /// First factor
    a: Vec<f32>,
    /// Second factor
    b: Vec<f32>,
    /// Shapes
    shape: (usize, usize, usize), // (rows, rank, cols)
}

/// Layer normalization
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LayerNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
    eps: f32,
}

impl FastGRNNRouter {
    /// Create a new router
    pub fn new(config: &RouterConfig) -> Result<Self> {
        let cell = FastGRNNCell::new(config.input_dim, config.hidden_dim, config.sparsity, config.rank);
        let output_heads = OutputHeads::new(config.hidden_dim);
        let input_norm = LayerNorm::new(config.input_dim);

        Ok(Self {
            cell,
            output_heads,
            input_norm,
            config: config.clone(),
        })
    }

    /// Load router from weights file
    pub fn load(path: impl AsRef<Path>, config: &RouterConfig) -> Result<Self> {
        let data = std::fs::read(path.as_ref())?;
        let (cell, output_heads, input_norm): (FastGRNNCell, OutputHeads, LayerNorm) =
            bincode::serde::decode_from_slice(&data, bincode::config::standard())
                .map_err(|e| Error::Serialization(e.to_string()))?
                .0;

        Ok(Self {
            cell,
            output_heads,
            input_norm,
            config: config.clone(),
        })
    }

    /// Save router weights
    pub fn save_weights(&self, path: impl AsRef<Path>) -> Result<()> {
        let data = bincode::serde::encode_to_vec(
            (&self.cell, &self.output_heads, &self.input_norm),
            bincode::config::standard(),
        ).map_err(|e| Error::Serialization(e.to_string()))?;

        std::fs::write(path, data)?;
        Ok(())
    }

    /// Forward pass through router
    pub fn forward(&self, features: &[f32], hidden: &[f32]) -> Result<RoutingDecision> {
        // Validate input dimensions
        if features.len() != self.config.input_dim {
            return Err(RouterError::InvalidFeatures {
                expected: self.config.input_dim,
                actual: features.len(),
            }.into());
        }

        // Normalize input
        let x = self.input_norm.forward(features);

        // FastGRNN cell
        let h_new = self.cell.forward(&x, hidden);

        // Output heads
        let model_logits = self.output_heads.model_forward(&h_new);
        let context_logits = self.output_heads.context_forward(&h_new);
        let temp_raw = self.output_heads.temp_forward(&h_new);
        let top_p_raw = self.output_heads.top_p_forward(&h_new);
        let conf_raw = self.output_heads.confidence_forward(&h_new);

        // Activations
        let model_probs = softmax(&model_logits);
        let context_probs = softmax(&context_logits);
        let temperature = sigmoid(temp_raw) * 2.0;
        let top_p = sigmoid(top_p_raw);
        let confidence = sigmoid(conf_raw);

        // Decode decisions
        let (model, context_size) = if confidence >= self.config.confidence_threshold {
            let model_idx = argmax(&model_probs);
            let context_idx = argmax(&context_probs);
            (ModelSize::from_index(model_idx), CONTEXT_BINS[context_idx])
        } else {
            // Safe defaults
            (ModelSize::B1_2, 2048)
        };

        Ok(RoutingDecision {
            model,
            context_size,
            temperature,
            top_p,
            confidence,
            model_probs: [model_probs[0], model_probs[1], model_probs[2], model_probs[3]],
            new_hidden: h_new,
            features: features.to_vec(),
        })
    }

    /// Get model parameters for training
    pub fn parameters(&self) -> Vec<&[f32]> {
        vec![
            &self.cell.w_z.values,
            &self.cell.w_h.values,
            &self.cell.b_z,
            &self.cell.b_h,
            &self.output_heads.w_model,
            &self.output_heads.w_context,
            &self.output_heads.w_temp,
            &self.output_heads.w_top_p,
            &self.output_heads.w_conf,
        ]
    }

    /// Get mutable parameters for training
    pub fn parameters_mut(&mut self) -> Vec<&mut [f32]> {
        vec![
            &mut self.cell.w_z.values,
            &mut self.cell.w_h.values,
            &mut self.cell.b_z,
            &mut self.cell.b_h,
            &mut self.output_heads.w_model,
            &mut self.output_heads.w_context,
            &mut self.output_heads.w_temp,
            &mut self.output_heads.w_top_p,
            &mut self.output_heads.w_conf,
        ]
    }
}

impl FastGRNNCell {
    fn new(input_dim: usize, hidden_dim: usize, sparsity: f32, rank: usize) -> Self {
        Self {
            w_z: SparseMatrix::random(hidden_dim, input_dim, sparsity),
            u_z: LowRankMatrix::random(hidden_dim, rank, hidden_dim),
            b_z: vec![0.0; hidden_dim],
            w_h: SparseMatrix::random(hidden_dim, input_dim, sparsity),
            u_h: LowRankMatrix::random(hidden_dim, rank, hidden_dim),
            b_h: vec![0.0; hidden_dim],
            zeta: 1.0,
            nu: 0.5,
        }
    }

    fn forward(&self, x: &[f32], h: &[f32]) -> Vec<f32> {
        // z = sigmoid(W_z @ x + U_z @ h + b_z)
        let z_pre: Vec<f32> = (0..self.b_z.len())
            .map(|i| {
                self.w_z.matvec_element(i, x)
                    + self.u_z.matvec_element(i, h)
                    + self.b_z[i]
            })
            .collect();
        let z: Vec<f32> = z_pre.iter().map(|&v| sigmoid(v)).collect();

        // h_tilde = tanh(W_h @ x + U_h @ h + b_h)
        let h_tilde_pre: Vec<f32> = (0..self.b_h.len())
            .map(|i| {
                self.w_h.matvec_element(i, x)
                    + self.u_h.matvec_element(i, h)
                    + self.b_h[i]
            })
            .collect();
        let h_tilde: Vec<f32> = h_tilde_pre.iter().map(|&v| v.tanh()).collect();

        // h_new = (zeta * (1 - z) + nu) * h_tilde + z * h
        z.iter()
            .zip(h_tilde.iter())
            .zip(h.iter())
            .map(|((&z_i, &ht_i), &h_i)| {
                (self.zeta * (1.0 - z_i) + self.nu) * ht_i + z_i * h_i
            })
            .collect()
    }
}

impl SparseMatrix {
    fn random(rows: usize, cols: usize, sparsity: f32) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let nnz = ((rows * cols) as f32 * (1.0 - sparsity)) as usize;

        let mut indices = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);

        for _ in 0..nnz {
            let row = rng.gen_range(0..rows);
            let col = rng.gen_range(0..cols);
            indices.push((row, col));
            values.push(rng.gen_range(-0.1..0.1));
        }

        Self {
            indices,
            values,
            shape: (rows, cols),
        }
    }

    fn matvec_element(&self, row: usize, x: &[f32]) -> f32 {
        self.indices.iter()
            .zip(self.values.iter())
            .filter(|((r, _), _)| *r == row)
            .map(|((_, c), v)| v * x.get(*c).unwrap_or(&0.0))
            .sum()
    }
}

impl LowRankMatrix {
    fn random(rows: usize, rank: usize, cols: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let a: Vec<f32> = (0..rows * rank)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();
        let b: Vec<f32> = (0..rank * cols)
            .map(|_| rng.gen_range(-0.1..0.1))
            .collect();

        Self {
            a,
            b,
            shape: (rows, rank, cols),
        }
    }

    fn matvec_element(&self, row: usize, x: &[f32]) -> f32 {
        let (_, rank, cols) = self.shape;

        // Compute (A @ B) @ x for the given row
        // = sum_k A[row, k] * sum_j B[k, j] * x[j]
        let mut result = 0.0;
        for k in 0..rank {
            let a_rk = self.a[row * rank + k];
            let mut b_dot_x = 0.0;
            for j in 0..cols.min(x.len()) {
                b_dot_x += self.b[k * cols + j] * x[j];
            }
            result += a_rk * b_dot_x;
        }
        result
    }
}

impl LayerNorm {
    fn new(dim: usize) -> Self {
        Self {
            gamma: vec![1.0; dim],
            beta: vec![0.0; dim],
            eps: 1e-5,
        }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mean = x.iter().sum::<f32>() / x.len() as f32;
        let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / x.len() as f32;
        let std = (var + self.eps).sqrt();

        x.iter()
            .zip(self.gamma.iter())
            .zip(self.beta.iter())
            .map(|((&xi, &g), &b)| g * (xi - mean) / std + b)
            .collect()
    }
}

impl OutputHeads {
    fn new(hidden_dim: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let random_vec = |size: usize| -> Vec<f32> {
            (0..size).map(|_| rng.gen_range(-0.1..0.1)).collect()
        };

        Self {
            w_model: random_vec(hidden_dim * 4),
            b_model: vec![0.0; 4],
            w_context: random_vec(hidden_dim * 5),
            b_context: vec![0.0; 5],
            w_temp: random_vec(hidden_dim),
            b_temp: 0.0,
            w_top_p: random_vec(hidden_dim),
            b_top_p: 0.0,
            w_conf: random_vec(hidden_dim),
            b_conf: 0.0,
        }
    }

    fn model_forward(&self, h: &[f32]) -> Vec<f32> {
        linear_forward(h, &self.w_model, &self.b_model, 4)
    }

    fn context_forward(&self, h: &[f32]) -> Vec<f32> {
        linear_forward(h, &self.w_context, &self.b_context, 5)
    }

    fn temp_forward(&self, h: &[f32]) -> f32 {
        dot(h, &self.w_temp) + self.b_temp
    }

    fn top_p_forward(&self, h: &[f32]) -> f32 {
        dot(h, &self.w_top_p) + self.b_top_p
    }

    fn confidence_forward(&self, h: &[f32]) -> f32 {
        dot(h, &self.w_conf) + self.b_conf
    }
}

// Helper functions

fn linear_forward(x: &[f32], w: &[f32], b: &[f32], out_dim: usize) -> Vec<f32> {
    let in_dim = x.len();
    (0..out_dim)
        .map(|i| {
            let mut sum = b[i];
            for j in 0..in_dim {
                sum += w[i * in_dim + j] * x[j];
            }
            sum
        })
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = x.iter().map(|v| (v - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|v| v / sum).collect()
}

fn argmax(x: &[f32]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let config = RouterConfig::default();
        let router = FastGRNNRouter::new(&config).unwrap();
        assert_eq!(router.config.input_dim, 128);
    }

    #[test]
    fn test_router_forward() {
        let config = RouterConfig::default();
        let router = FastGRNNRouter::new(&config).unwrap();

        let features = vec![0.5f32; config.input_dim];
        let hidden = vec![0.0f32; config.hidden_dim];

        let decision = router.forward(&features, &hidden).unwrap();

        // Verify outputs are valid
        assert!(decision.temperature >= 0.0 && decision.temperature <= 2.0);
        assert!(decision.top_p >= 0.0 && decision.top_p <= 1.0);
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
        assert_eq!(decision.new_hidden.len(), config.hidden_dim);
    }

    #[test]
    fn test_softmax() {
        let x = vec![1.0, 2.0, 3.0];
        let result = softmax(&x);
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_layer_norm() {
        let norm = LayerNorm::new(4);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let result = norm.forward(&x);
        let mean: f32 = result.iter().sum::<f32>() / result.len() as f32;
        assert!(mean.abs() < 0.01); // Should be close to 0
    }
}
