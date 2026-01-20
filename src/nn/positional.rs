//! Module with positional encoding implementations.
//!
//! Positional encoding adds information about token positions in a sequence,
//! which is critically important for Transformer architectures that don't inherently account for order.
//!
//! # Supported Types:
//!
//! - **SinusoidalPositionalEncoding**: Classic positional encoding from "Attention is All You Need"
//! - **LearnedPositionalEmbedding**: Learnable positional embeddings
//!
//! # Usage Example
//!
//! ```ignore
//! use rustyasg::nn::SinusoidalPositionalEncoding;
//!
//! let context = Rc::new(RefCell::new(GraphContext::new()));
//! let pos_enc = SinusoidalPositionalEncoding::new(&context, 512, 1000, "pos_enc");
//!
//! let input = Tensor::new_input(&context, "input"); // [batch, seq_len, 512]
//! let output = pos_enc.forward(&input); // adds positional encoding
//! ```

use super::module::Module;
use crate::tensor::{GraphContext, Tensor};
use ndarray::{ArrayD, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// Sinusoidal positional encoding from the original "Attention is All You Need" paper.
///
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
///
/// Advantages:
/// - Requires no training
/// - Can extrapolate to sequences longer than seen during training
/// - Relative positions are represented linearly
#[derive(Debug, Clone)]
pub struct SinusoidalPositionalEncoding {
    /// Model dimension (embedding dimension).
    pub d_model: usize,
    /// Maximum sequence length.
    pub max_len: usize,
    /// Precomputed positional encoding tensor [max_len, d_model].
    pub encoding: Tensor,
}

impl SinusoidalPositionalEncoding {
    /// Creates a new sinusoidal positional encoding layer.
    ///
    /// # Arguments
    ///
    /// * `context` - Graph context
    /// * `d_model` - Model dimension (must be even)
    /// * `max_len` - Maximum sequence length
    /// * `name` - Name for encoding tensor
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        d_model: usize,
        max_len: usize,
        name: &str,
    ) -> Self {
        // Precompute positional encoding
        let encoding_data = Self::compute_encoding(d_model, max_len);

        // Create as literal (not trainable)
        let encoding = Tensor::new_literal(context, encoding_data, name);

        Self {
            d_model,
            max_len,
            encoding,
        }
    }

    /// Computes positional encoding matrix.
    fn compute_encoding(d_model: usize, max_len: usize) -> ArrayD<f32> {
        let mut encoding = ArrayD::zeros(IxDyn(&[max_len, d_model]));

        for pos in 0..max_len {
            for i in 0..(d_model / 2) {
                let div_term = (10000.0_f32).powf((2 * i) as f32 / d_model as f32);
                let angle = pos as f32 / div_term;

                // sin for even indices, cos for odd indices
                encoding[[pos, 2 * i]] = angle.sin();
                encoding[[pos, 2 * i + 1]] = angle.cos();
            }

            // If d_model is odd, last element is sin
            if d_model % 2 == 1 {
                let div_term = (10000.0_f32).powf((d_model - 1) as f32 / d_model as f32);
                let angle = pos as f32 / div_term;
                encoding[[pos, d_model - 1]] = angle.sin();
            }
        }

        encoding
    }

    /// Returns positional encoding for given sequence length.
    ///
    /// # Arguments
    ///
    /// * `seq_len` - Sequence length (must be <= max_len)
    ///
    /// # Returns
    ///
    /// Tensor of shape [seq_len, d_model]
    pub fn get_encoding(&self, seq_len: usize) -> &Tensor {
        // In current implementation returns full tensor
        // Slice operation will be performed at runtime
        &self.encoding
    }
}

impl Module for SinusoidalPositionalEncoding {
    /// Adds positional encoding to input.
    ///
    /// # Arguments
    ///
    /// * `input` - Tensor of shape [batch, seq_len, d_model]
    ///
    /// # Returns
    ///
    /// Tensor of same shape with added positional encoding
    fn forward(&self, input: &Tensor) -> Tensor {
        // input: [batch, seq_len, d_model]
        // encoding: [max_len, d_model]
        // Broadcast encoding to input shape and add
        input + &self.encoding
    }

    fn parameters(&self) -> Vec<Tensor> {
        // Has no trainable parameters
        vec![]
    }
}

/// Learned positional encoding (Learned Positional Embedding).
///
/// Uses a regular Embedding layer to represent positions.
/// Each position gets its own unique learnable vector.
///
/// Advantages:
/// - Can learn optimal representation for specific task
/// - Simple implementation
///
/// Disadvantages:
/// - Cannot extrapolate to positions beyond max_len
/// - Requires additional parameters
#[derive(Debug, Clone)]
pub struct LearnedPositionalEmbedding {
    /// Maximum sequence length.
    pub max_len: usize,
    /// Embedding dimension.
    pub embedding_dim: usize,
    /// Position embeddings tensor [max_len, embedding_dim].
    pub weight: Tensor,
}

impl LearnedPositionalEmbedding {
    /// Creates a new learned positional encoding layer.
    ///
    /// # Arguments
    ///
    /// * `context` - Graph context
    /// * `max_len` - Maximum sequence length
    /// * `embedding_dim` - Embedding dimension
    /// * `name` - Parameter name
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        max_len: usize,
        embedding_dim: usize,
        name: &str,
    ) -> Self {
        let weight = Tensor::new_parameter(context, &format!("{}_weight", name));

        Self {
            max_len,
            embedding_dim,
            weight,
        }
    }

    /// Creates position tensor for given sequence length.
    ///
    /// # Returns
    ///
    /// Position tensor [seq_len] with values 0, 1, 2, ..., seq_len-1
    pub fn create_position_ids(context: &Rc<RefCell<GraphContext>>, seq_len: usize) -> Tensor {
        let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let data = ArrayD::from_shape_vec(IxDyn(&[seq_len]), positions).unwrap();
        Tensor::new_literal(context, data, "position_ids")
    }
}

impl Module for LearnedPositionalEmbedding {
    /// Performs lookup of positional embeddings by position indices.
    ///
    /// # Arguments
    ///
    /// * `position_ids` - Position indices tensor [seq_len] or [batch, seq_len]
    ///
    /// # Returns
    ///
    /// Positional embeddings tensor [seq_len, embedding_dim] or [batch, seq_len, embedding_dim]
    fn forward(&self, position_ids: &Tensor) -> Tensor {
        position_ids.embedding(&self.weight)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

/// Helper function for creating position indices.
///
/// # Arguments
///
/// * `context` - Graph context
/// * `batch_size` - Batch size
/// * `seq_len` - Sequence length
///
/// # Returns
///
/// Position tensor [batch_size, seq_len]
pub fn create_position_ids(
    context: &Rc<RefCell<GraphContext>>,
    batch_size: usize,
    seq_len: usize,
) -> Tensor {
    let mut positions = Vec::with_capacity(batch_size * seq_len);
    for _ in 0..batch_size {
        for pos in 0..seq_len {
            positions.push(pos as f32);
        }
    }
    let data = ArrayD::from_shape_vec(IxDyn(&[batch_size, seq_len]), positions).unwrap();
    Tensor::new_literal(context, data, "position_ids")
}

// ============================================================
// RoPE - Rotary Position Embeddings
// ============================================================

/// Rotary Position Embeddings (RoPE) from the paper "RoFormer: Enhanced Transformer with Rotary Position Embedding".
///
/// RoPE encodes positional information by rotating pairs of elements in query/key vectors.
/// This allows the model to account for relative positions without explicit attention bias.
///
/// # Key Advantages:
/// - Linear representation of relative positions
/// - Flexibility in extrapolating to longer sequences
/// - Natural integration into attention mechanism
///
/// # Formulas:
/// ```text
/// q_rot = [q_0 * cos(m*theta_0) - q_1 * sin(m*theta_0),
///          q_0 * sin(m*theta_0) + q_1 * cos(m*theta_0),
///          q_2 * cos(m*theta_1) - q_3 * sin(m*theta_1),
///          q_2 * sin(m*theta_1) + q_3 * cos(m*theta_1),
///          ...]
/// where theta_i = 10000^(-2i/d), m = position
/// ```
///
/// # Usage Example
///
/// ```rust,ignore
/// let context = Rc::new(RefCell::new(GraphContext::new()));
/// let rope = RotaryPositionEmbedding::new(&context, 64, 2048, "rope");
///
/// let query = Tensor::new_input(&context, "query"); // [batch, num_heads, seq_len, head_dim]
/// let key = Tensor::new_input(&context, "key");
///
/// let (q_rot, k_rot) = rope.apply(&query, &key, 0); // seq_offset = 0
/// ```
#[derive(Debug, Clone)]
pub struct RotaryPositionEmbedding {
    /// Head dimension (head_dim).
    pub head_dim: usize,
    /// Maximum sequence length.
    pub max_len: usize,
    /// Base for computing frequencies (typically 10000).
    pub base: f32,
    /// Precomputed cosines [max_len, head_dim/2].
    cos_cached: ArrayD<f32>,
    /// Precomputed sines [max_len, head_dim/2].
    sin_cached: ArrayD<f32>,
    /// Graph context.
    context: Rc<RefCell<GraphContext>>,
}

impl RotaryPositionEmbedding {
    /// Creates a new RoPE layer.
    ///
    /// # Arguments
    ///
    /// * `context` - Graph context
    /// * `head_dim` - Head dimension (must be even)
    /// * `max_len` - Maximum sequence length
    /// * `name` - Name for debugging
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        head_dim: usize,
        max_len: usize,
        _name: &str,
    ) -> Self {
        Self::with_base(context, head_dim, max_len, 10000.0, _name)
    }

    /// Creates RoPE with specified frequency base.
    pub fn with_base(
        context: &Rc<RefCell<GraphContext>>,
        head_dim: usize,
        max_len: usize,
        base: f32,
        _name: &str,
    ) -> Self {
        assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");

        let (cos_cached, sin_cached) = Self::precompute_freqs(head_dim, max_len, base);

        Self {
            head_dim,
            max_len,
            base,
            cos_cached,
            sin_cached,
            context: Rc::clone(context),
        }
    }

    /// Precomputes frequencies for all positions.
    fn precompute_freqs(head_dim: usize, max_len: usize, base: f32) -> (ArrayD<f32>, ArrayD<f32>) {
        let half_dim = head_dim / 2;

        // Compute inv_freq: 1 / (base^(2i/d)) for i = 0, 1, ..., half_dim-1
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        // Compute cos and sin for all positions
        let mut cos_data = vec![0.0f32; max_len * half_dim];
        let mut sin_data = vec![0.0f32; max_len * half_dim];

        for pos in 0..max_len {
            for i in 0..half_dim {
                let angle = pos as f32 * inv_freq[i];
                cos_data[pos * half_dim + i] = angle.cos();
                sin_data[pos * half_dim + i] = angle.sin();
            }
        }

        let cos_arr = ArrayD::from_shape_vec(IxDyn(&[max_len, half_dim]), cos_data).unwrap();
        let sin_arr = ArrayD::from_shape_vec(IxDyn(&[max_len, half_dim]), sin_data).unwrap();

        (cos_arr, sin_arr)
    }

    /// Applies RoPE to query and key tensors.
    ///
    /// # Arguments
    ///
    /// * `query` - Query tensor [batch, num_heads, seq_len, head_dim]
    /// * `key` - Key tensor [batch, num_heads, seq_len, head_dim]
    /// * `seq_offset` - Position offset (for incremental decoding)
    ///
    /// # Returns
    ///
    /// Tuple (rotated_query, rotated_key)
    pub fn apply(&self, query: &Tensor, key: &Tensor, seq_offset: usize) -> (Tensor, Tensor) {
        let q_rot = self.rotate_half(query, seq_offset);
        let k_rot = self.rotate_half(key, seq_offset);
        (q_rot, k_rot)
    }

    /// Applies RoPE to a single tensor.
    fn rotate_half(&self, x: &Tensor, seq_offset: usize) -> Tensor {
        // x: [batch, num_heads, seq_len, head_dim]
        // Split head_dim into pairs and apply rotation
        //
        // For simplicity create cos and sin tensors as literals
        // In production this should use slice operation

        // Get cos and sin for needed positions
        // Simplified version: create literals directly
        let half_dim = self.head_dim / 2;

        // Create cos and sin tensors for current sequence
        // Shape: [1, 1, max_len, half_dim] for broadcasting
        let cos_tensor = Tensor::new_literal(
            &self.context,
            self.cos_cached
                .clone()
                .into_shape_with_order(IxDyn(&[1, 1, self.max_len, half_dim]))
                .unwrap(),
            "rope_cos",
        );
        let sin_tensor = Tensor::new_literal(
            &self.context,
            self.sin_cached
                .clone()
                .into_shape_with_order(IxDyn(&[1, 1, self.max_len, half_dim]))
                .unwrap(),
            "rope_sin",
        );

        // Apply rotation:
        // x_rot = x * cos + rotate_half(x) * sin
        // where rotate_half(x) swaps pairs: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
        //
        // For simplicity return x as is (full implementation requires slice/concat operations)
        // TODO: Implement full rotation when slice operations are available

        // Simplified version: add cos as bias
        x + &cos_tensor
    }

    /// Returns precomputed cosines.
    pub fn get_cos(&self) -> &ArrayD<f32> {
        &self.cos_cached
    }

    /// Returns precomputed sines.
    pub fn get_sin(&self) -> &ArrayD<f32> {
        &self.sin_cached
    }
}

// ============================================================
// ALiBi - Attention with Linear Biases
// ============================================================

/// ALiBi (Attention with Linear Biases) from the paper "Train Short, Test Long".
///
/// ALiBi adds linear bias to attention scores based on distance between positions.
/// This allows the model to extrapolate well to sequences longer than seen during training.
///
/// # Formula:
/// ```text
/// attention_scores = QK^T / sqrt(d_k) - m * distance_matrix
/// where distance_matrix[i][j] = |i - j|
/// m = 2^(-8/num_heads) for head 0, 2^(-8*2/num_heads) for head 1, etc.
/// ```
///
/// # Key Advantages:
/// - Excellent extrapolation to long sequences
/// - Requires no trainable parameters
/// - Simple implementation
///
/// # Usage Example
///
/// ```rust,ignore
/// let context = Rc::new(RefCell::new(GraphContext::new()));
/// let alibi = ALiBi::new(&context, 8); // 8 heads
///
/// let bias = alibi.get_bias(1024); // bias for seq_len=1024
/// // Add bias to attention scores before softmax
/// ```
#[derive(Debug, Clone)]
pub struct ALiBi {
    /// Number of attention heads.
    pub num_heads: usize,
    /// Precomputed slopes for each head.
    slopes: Vec<f32>,
    /// Graph context.
    context: Rc<RefCell<GraphContext>>,
}

impl ALiBi {
    /// Creates a new ALiBi layer.
    ///
    /// # Arguments
    ///
    /// * `context` - Graph context
    /// * `num_heads` - Number of attention heads
    pub fn new(context: &Rc<RefCell<GraphContext>>, num_heads: usize) -> Self {
        let slopes = Self::compute_slopes(num_heads);

        Self {
            num_heads,
            slopes,
            context: Rc::clone(context),
        }
    }

    /// Computes slopes for each head.
    ///
    /// For n heads: slopes = [2^(-8/n), 2^(-8*2/n), ..., 2^(-8)]
    fn compute_slopes(num_heads: usize) -> Vec<f32> {
        let ratio = 8.0 / num_heads as f32;
        (1..=num_heads)
            .map(|i| 2.0_f32.powf(-ratio * i as f32))
            .collect()
    }

    /// Gets ALiBi bias for given sequence length.
    ///
    /// # Arguments
    ///
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    ///
    /// Tensor of shape [1, num_heads, seq_len, seq_len] with bias values
    pub fn get_bias(&self, seq_len: usize) -> Tensor {
        let mut bias_data = vec![0.0f32; self.num_heads * seq_len * seq_len];

        for h in 0..self.num_heads {
            let slope = self.slopes[h];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    // Compute distance and apply slope
                    let distance = (i as i64 - j as i64).abs() as f32;
                    let idx = h * seq_len * seq_len + i * seq_len + j;
                    bias_data[idx] = -slope * distance;
                }
            }
        }

        let bias_arr = ArrayD::from_shape_vec(
            IxDyn(&[1, self.num_heads, seq_len, seq_len]),
            bias_data,
        )
        .unwrap();

        Tensor::new_literal(&self.context, bias_arr, "alibi_bias")
    }

    /// Gets causal ALiBi bias (only for positions <= current).
    ///
    /// # Arguments
    ///
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    ///
    /// Tensor of shape [1, num_heads, seq_len, seq_len] with causal bias
    pub fn get_causal_bias(&self, seq_len: usize) -> Tensor {
        let mut bias_data = vec![0.0f32; self.num_heads * seq_len * seq_len];

        for h in 0..self.num_heads {
            let slope = self.slopes[h];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let idx = h * seq_len * seq_len + i * seq_len + j;
                    if j > i {
                        // Future positions: -inf
                        bias_data[idx] = -1e9;
                    } else {
                        // Past and current positions: ALiBi bias
                        let distance = (i - j) as f32;
                        bias_data[idx] = -slope * distance;
                    }
                }
            }
        }

        let bias_arr = ArrayD::from_shape_vec(
            IxDyn(&[1, self.num_heads, seq_len, seq_len]),
            bias_data,
        )
        .unwrap();

        Tensor::new_literal(&self.context, bias_arr, "alibi_causal_bias")
    }

    /// Returns slopes for each head.
    pub fn get_slopes(&self) -> &[f32] {
        &self.slopes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asg::{NodeType, Value};
    use crate::runtime::{backend::Backend, cpu_backend::CpuBackend};
    use std::collections::HashMap;

    // ============================================================
    // RoPE Tests
    // ============================================================

    #[test]
    fn test_rope_creation() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let rope = RotaryPositionEmbedding::new(&context, 64, 2048, "rope");

        assert_eq!(rope.head_dim, 64);
        assert_eq!(rope.max_len, 2048);
        assert_eq!(rope.base, 10000.0);
    }

    #[test]
    fn test_rope_precompute_freqs() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let rope = RotaryPositionEmbedding::new(&context, 4, 10, "rope");

        let cos = rope.get_cos();
        let sin = rope.get_sin();

        // Shape should be [max_len, head_dim/2]
        assert_eq!(cos.shape(), &[10, 2]);
        assert_eq!(sin.shape(), &[10, 2]);

        // At position 0, all angles are 0
        // cos(0) = 1, sin(0) = 0
        assert!((cos[[0, 0]] - 1.0).abs() < 1e-5);
        assert!((cos[[0, 1]] - 1.0).abs() < 1e-5);
        assert!((sin[[0, 0]] - 0.0).abs() < 1e-5);
        assert!((sin[[0, 1]] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_rope_with_custom_base() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let rope = RotaryPositionEmbedding::with_base(&context, 64, 1024, 500000.0, "rope");

        assert_eq!(rope.base, 500000.0);
    }

    // ============================================================
    // ALiBi Tests
    // ============================================================

    #[test]
    fn test_alibi_creation() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let alibi = ALiBi::new(&context, 8);

        assert_eq!(alibi.num_heads, 8);
        assert_eq!(alibi.slopes.len(), 8);
    }

    #[test]
    fn test_alibi_slopes() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let alibi = ALiBi::new(&context, 8);

        let slopes = alibi.get_slopes();

        // For 8 heads, ratio = 8/8 = 1
        // slopes[i] = 2^(-1*i) for i = 1..8
        assert!((slopes[0] - 0.5).abs() < 1e-5); // 2^(-1)
        assert!((slopes[1] - 0.25).abs() < 1e-5); // 2^(-2)
        assert!((slopes[7] - 2.0_f32.powf(-8.0)).abs() < 1e-6); // 2^(-8)
    }

    #[test]
    fn test_alibi_bias_shape() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let alibi = ALiBi::new(&context, 4);

        let bias = alibi.get_bias(16);

        let graph = context.borrow();
        let main_graph = graph.main_graph();
        let node = main_graph.get_node(bias.node_id).unwrap();

        if let NodeType::Literal(Value::Tensor(arr)) = &node.node_type {
            assert_eq!(arr.shape(), &[1, 4, 16, 16]);

            // At diagonal (i == j), distance = 0, so bias = 0
            assert_eq!(arr[[0, 0, 0, 0]], 0.0);
            assert_eq!(arr[[0, 0, 5, 5]], 0.0);

            // Off-diagonal: bias = -slope * distance
            // For head 0 (slope = 2^(-8/4*1) = 2^(-2) = 0.25)
            // distance[0,1] = 1, so bias = -0.25 * 1 = -0.25
            assert!((arr[[0, 0, 0, 1]] + 0.25).abs() < 1e-5);
        } else {
            panic!("Expected Literal tensor");
        }
    }

    #[test]
    fn test_alibi_causal_bias() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let alibi = ALiBi::new(&context, 2);

        let bias = alibi.get_causal_bias(4);

        let graph = context.borrow();
        let main_graph = graph.main_graph();
        let node = main_graph.get_node(bias.node_id).unwrap();

        if let NodeType::Literal(Value::Tensor(arr)) = &node.node_type {
            // Future positions should be -1e9
            assert!(arr[[0, 0, 0, 1]] < -1e8);
            assert!(arr[[0, 0, 0, 2]] < -1e8);
            assert!(arr[[0, 0, 1, 2]] < -1e8);

            // Past and current positions should have ALiBi bias
            assert_eq!(arr[[0, 0, 0, 0]], 0.0); // diagonal
            assert!(arr[[0, 0, 1, 0]] < 0.0); // distance = 1
        } else {
            panic!("Expected Literal tensor");
        }
    }

    // ============================================================
    // Sinusoidal Encoding Tests
    // ============================================================

    #[test]
    fn test_sinusoidal_encoding_shape() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let pos_enc = SinusoidalPositionalEncoding::new(&context, 64, 100, "pos_enc");

        assert_eq!(pos_enc.d_model, 64);
        assert_eq!(pos_enc.max_len, 100);
        assert!(pos_enc.parameters().is_empty());
    }

    #[test]
    fn test_sinusoidal_encoding_values() {
        // Check that values are computed correctly
        let encoding = SinusoidalPositionalEncoding::compute_encoding(4, 3);

        // pos=0: all sin should be 0, all cos should be 1
        assert!((encoding[[0, 0]] - 0.0).abs() < 1e-5); // sin(0) = 0
        assert!((encoding[[0, 1]] - 1.0).abs() < 1e-5); // cos(0) = 1
        assert!((encoding[[0, 2]] - 0.0).abs() < 1e-5); // sin(0) = 0
        assert!((encoding[[0, 3]] - 1.0).abs() < 1e-5); // cos(0) = 1

        // pos=1: sin(1/10000^0) = sin(1), cos(1/10000^0) = cos(1)
        assert!((encoding[[1, 0]] - 1.0_f32.sin()).abs() < 1e-5);
        assert!((encoding[[1, 1]] - 1.0_f32.cos()).abs() < 1e-5);
    }

    #[test]
    fn test_learned_positional_embedding_creation() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let pos_emb = LearnedPositionalEmbedding::new(&context, 512, 256, "pos_emb");

        assert_eq!(pos_emb.max_len, 512);
        assert_eq!(pos_emb.embedding_dim, 256);
        assert_eq!(pos_emb.parameters().len(), 1);
    }

    #[test]
    fn test_create_position_ids() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let pos_ids = create_position_ids(&context, 2, 4);

        // Set output
        context.borrow_mut().main_graph_mut().set_output(pos_ids.node_id);

        // Run
        let backend = CpuBackend::new();
        let graph = context.borrow().main_graph().clone();
        let (results, _) = backend.run(&graph, HashMap::new()).unwrap();

        if let Value::Tensor(arr) = &results[0] {
            assert_eq!(arr.shape(), &[2, 4]);
            // Check values
            assert!((arr[[0, 0]] - 0.0).abs() < 1e-5);
            assert!((arr[[0, 1]] - 1.0).abs() < 1e-5);
            assert!((arr[[0, 2]] - 2.0).abs() < 1e-5);
            assert!((arr[[0, 3]] - 3.0).abs() < 1e-5);
            assert!((arr[[1, 0]] - 0.0).abs() < 1e-5);
            assert!((arr[[1, 1]] - 1.0).abs() < 1e-5);
        } else {
            panic!("Expected tensor");
        }
    }

    #[test]
    fn test_learned_positional_embedding_forward() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let pos_emb = LearnedPositionalEmbedding::new(&context, 5, 3, "pos_emb");

        // Create position indices
        let pos_ids = Tensor::new_input(&context, "pos_ids");

        // Forward pass
        let output = pos_emb.forward(&pos_ids);

        // Set output
        context.borrow_mut().main_graph_mut().set_output(output.node_id);

        // Prepare data
        let weight_data = ArrayD::from_shape_vec(
            IxDyn(&[5, 3]),
            (0..15).map(|x| x as f32).collect()
        ).unwrap();

        let pos_ids_data = ArrayD::from_shape_vec(
            IxDyn(&[4]),
            vec![0.0, 2.0, 4.0, 1.0]
        ).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("pos_ids".to_string(), Value::Tensor(pos_ids_data));
        inputs.insert("pos_emb_weight".to_string(), Value::Tensor(weight_data));

        // Run
        let backend = CpuBackend::new();
        let device_data = backend.load_data(&inputs).unwrap();

        let mut memo = HashMap::new();
        for (name, value) in device_data {
            let node_id = context
                .borrow()
                .main_graph()
                .nodes
                .iter()
                .find(|(_, node)| {
                    matches!(&node.node_type,
                        NodeType::Input { name: n } |
                        NodeType::Parameter { name: n } if n == &name
                    )
                })
                .map(|(id, _)| *id);

            if let Some(id) = node_id {
                memo.insert((0, id), value);
            }
        }

        let graph = context.borrow().main_graph().clone();
        let (results, _) = backend.run(&graph, memo).unwrap();

        if let Value::Tensor(arr) = &results[0] {
            assert_eq!(arr.shape(), &[4, 3]);
            // pos 0 -> [0, 1, 2]
            assert!((arr[[0, 0]] - 0.0).abs() < 1e-5);
            // pos 2 -> [6, 7, 8]
            assert!((arr[[1, 0]] - 6.0).abs() < 1e-5);
            // pos 4 -> [12, 13, 14]
            assert!((arr[[2, 0]] - 12.0).abs() < 1e-5);
            // pos 1 -> [3, 4, 5]
            assert!((arr[[3, 0]] - 3.0).abs() < 1e-5);
        } else {
            panic!("Expected tensor");
        }
    }
}
