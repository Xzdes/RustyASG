//! Module implementing Multi-Head Attention
//! with mask support (causal, padding) for graph architecture.
//!
//! # Usage Example
//!
//! ```rust,ignore
//! use rustyasg::nn::{MultiHeadAttention, Module};
//! use rustyasg::tensor::{GraphContext, Tensor};
//!
//! let context = Rc::new(RefCell::new(GraphContext::new()));
//! let mha = MultiHeadAttention::new(&context, 512, 8, "mha");
//!
//! let input = Tensor::new_input(&context, "input"); // [batch, seq_len, 512]
//! let output = mha.forward(&input); // Self-attention
//!
//! // With causal mask (for decoder)
//! let output = mha.forward_with_mask(&input, &input, &input, Some(&causal_mask), None);
//! ```

use crate::asg::{NodeType, Value};
use crate::nn::{Linear, Module};
use crate::tensor::{GraphContext, Tensor};
use ndarray::{arr0, ArrayD, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// Attention mask type.
#[derive(Debug, Clone)]
pub enum AttentionMask {
    /// Causal mask - prevents attention to future positions.
    /// Automatically created based on seq_len.
    Causal,
    /// Padding mask - indicates which positions are padding.
    /// Tensor of shape [batch_size, seq_len], where 0 = padding, 1 = valid.
    Padding(Tensor),
    /// Arbitrary mask, passed directly.
    /// Tensor of shape [batch_size, 1, seq_len, seq_len] or [1, 1, seq_len, seq_len].
    Custom(Tensor),
}

/// Multi-Head Attention configuration.
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionConfig {
    /// Embedding dimension.
    pub embed_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dropout probability (not used yet).
    pub dropout: f32,
    /// Whether to use bias in projections.
    pub bias: bool,
}

impl Default for MultiHeadAttentionConfig {
    fn default() -> Self {
        Self {
            embed_dim: 512,
            num_heads: 8,
            dropout: 0.0,
            bias: true,
        }
    }
}

impl MultiHeadAttentionConfig {
    /// Creates configuration with specified parameters.
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        Self {
            embed_dim,
            num_heads,
            ..Default::default()
        }
    }

    /// Sets dropout.
    pub fn with_dropout(mut self, dropout: f32) -> Self {
        self.dropout = dropout;
        self
    }

    /// Disables bias in projections.
    pub fn without_bias(mut self) -> Self {
        self.bias = false;
        self
    }
}

/// Multi-Head Attention implementation.
///
/// Supports:
/// - Self-attention (Q = K = V)
/// - Cross-attention (Q != K = V)
/// - Causal masks (for decoders)
/// - Padding masks
/// - Arbitrary attention masks
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    scale: f32,
    context: Rc<RefCell<GraphContext>>,
}

impl MultiHeadAttention {
    /// Creates a new MultiHeadAttention layer.
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        embed_dim: usize,
        num_heads: usize,
        name: &str,
    ) -> Self {
        assert!(
            embed_dim % num_heads == 0,
            "embed_dim ({}) must be divisible by num_heads ({}) without remainder.",
            embed_dim,
            num_heads
        );
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let w_q_name = format!("{}.w_q", name);
        let w_k_name = format!("{}.w_k", name);
        let w_v_name = format!("{}.w_v", name);
        let w_o_name = format!("{}.w_o", name);

        Self {
            num_heads,
            head_dim,
            embed_dim,
            w_q: Linear::new(context, &w_q_name),
            w_k: Linear::new(context, &w_k_name),
            w_v: Linear::new(context, &w_v_name),
            w_o: Linear::new(context, &w_o_name),
            scale,
            context: Rc::clone(context),
        }
    }

    /// Creates MultiHeadAttention from configuration.
    pub fn from_config(
        context: &Rc<RefCell<GraphContext>>,
        config: MultiHeadAttentionConfig,
        name: &str,
    ) -> Self {
        Self::new(context, config.embed_dim, config.num_heads, name)
    }

    /// Scaled Dot-Product Attention.
    ///
    /// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
    ///
    /// # Arguments
    /// * `query` - Query tensor [batch, num_heads, seq_q, head_dim]
    /// * `key` - Key tensor [batch, num_heads, seq_k, head_dim]
    /// * `value` - Value tensor [batch, num_heads, seq_k, head_dim]
    /// * `mask` - Optional attention mask
    pub fn scaled_dot_product_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: Option<&Tensor>,
    ) -> Tensor {
        // QK^T: [batch, num_heads, seq_q, head_dim] @ [batch, num_heads, head_dim, seq_k]
        // -> [batch, num_heads, seq_q, seq_k]
        let k_transposed = key.transpose(2, 3);
        let scores = query.dot(&k_transposed);

        // Scale
        let scale_tensor = Tensor::new_literal(
            &self.context,
            arr0(self.scale).into_dyn(),
            "scale",
        );
        let scores_scaled = &scores * &scale_tensor;

        // Apply mask (if provided)
        let scores_masked = if let Some(m) = mask {
            // Mask: 0 for valid positions, -inf for masked positions
            // scores + mask
            &scores_scaled + m
        } else {
            scores_scaled
        };

        // Softmax along last dimension (seq_k)
        let attention_weights = scores_masked.softmax();

        // Apply attention to values: [batch, num_heads, seq_q, seq_k] @ [batch, num_heads, seq_k, head_dim]
        // -> [batch, num_heads, seq_q, head_dim]
        attention_weights.dot(value)
    }

    /// Forward pass with explicit Q, K, V and masks.
    ///
    /// # Arguments
    /// * `query` - Query tensor [batch, seq_q, embed_dim]
    /// * `key` - Key tensor [batch, seq_k, embed_dim]
    /// * `value` - Value tensor [batch, seq_k, embed_dim]
    /// * `attn_mask` - Optional attention mask
    /// * `key_padding_mask` - Optional key padding mask [batch, seq_k]
    pub fn forward_qkv(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
        key_padding_mask: Option<&Tensor>,
    ) -> Tensor {
        // Project Q, K, V
        let q = self.w_q.forward(query);
        let k = self.w_k.forward(key);
        let v = self.w_v.forward(value);

        // Reshape to [batch, seq, num_heads, head_dim] then transpose to [batch, num_heads, seq, head_dim]
        // For simplicity assuming batch_size and seq_len are known through shape
        let q_heads = self.split_heads_dynamic(&q);
        let k_heads = self.split_heads_dynamic(&k);
        let v_heads = self.split_heads_dynamic(&v);

        // Combine attention mask and key padding mask
        let combined_mask = self.combine_masks(attn_mask, key_padding_mask);

        // Scaled dot-product attention
        let attention_output = self.scaled_dot_product_attention(
            &q_heads,
            &k_heads,
            &v_heads,
            combined_mask.as_ref(),
        );

        // Combine heads: [batch, num_heads, seq_q, head_dim] -> [batch, seq_q, embed_dim]
        let concatenated = self.combine_heads_dynamic(&attention_output);

        // Output projection
        self.w_o.forward(&concatenated)
    }

    /// Helper function to split tensor into heads (dynamic version).
    fn split_heads_dynamic(&self, x: &Tensor) -> Tensor {
        // [batch, seq, embed_dim] -> [batch, seq, num_heads, head_dim]
        // -> [batch, num_heads, seq, head_dim]
        // Using reshape with -1 for automatic dimension inference
        // For now using simplified reshape for single batch
        x.reshape(vec![-1, -1, self.num_heads as i64, self.head_dim as i64])
            .transpose(1, 2)
    }

    /// Helper function to merge heads (dynamic version).
    fn combine_heads_dynamic(&self, x: &Tensor) -> Tensor {
        // [batch, num_heads, seq, head_dim] -> [batch, seq, num_heads, head_dim]
        // -> [batch, seq, embed_dim]
        x.transpose(1, 2)
            .reshape(vec![-1, -1, self.embed_dim as i64])
    }

    /// Combines attention mask and key padding mask.
    fn combine_masks(
        &self,
        attn_mask: Option<&Tensor>,
        key_padding_mask: Option<&Tensor>,
    ) -> Option<Tensor> {
        match (attn_mask, key_padding_mask) {
            (None, None) => None,
            (Some(m), None) => Some(m.clone()),
            (None, Some(kpm)) => {
                // Convert key_padding_mask [batch, seq_k] to [batch, 1, 1, seq_k]
                // and convert: 0 (valid) -> 0.0, 1 (padding) -> -inf
                Some(self.expand_padding_mask(kpm))
            }
            (Some(am), Some(kpm)) => {
                // Combine both masks
                let expanded_kpm = self.expand_padding_mask(kpm);
                Some(&am.clone() + &expanded_kpm)
            }
        }
    }

    /// Expands padding mask to attention scores shape.
    fn expand_padding_mask(&self, mask: &Tensor) -> Tensor {
        // mask: [batch, seq_k], value 0 = padding, 1 = valid
        // output: [batch, 1, 1, seq_k], value 0 for valid, -inf for padding
        // (1 - mask) * -1e9
        let one = Tensor::scalar(&self.context, 1.0);
        let neg_inf = Tensor::scalar(&self.context, -1e9);
        let inverted = &one - mask;
        &inverted * &neg_inf
    }

    /// Creates causal mask (upper triangular with -inf).
    ///
    /// # Arguments
    /// * `seq_len` - Sequence length
    ///
    /// # Returns
    /// Tensor of shape [1, 1, seq_len, seq_len]
    pub fn create_causal_mask(&self, seq_len: usize) -> Tensor {
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    mask_data[i * seq_len + j] = -1e9;
                }
            }
        }
        let mask_arr = ArrayD::from_shape_vec(IxDyn(&[1, 1, seq_len, seq_len]), mask_data).unwrap();
        Tensor::new_literal(&self.context, mask_arr, "causal_mask")
    }

    /// Creates padding mask from sequence lengths.
    ///
    /// # Arguments
    /// * `lengths` - Vector of lengths for each batch element
    /// * `max_len` - Maximum length (seq dimension size)
    ///
    /// # Returns
    /// Tensor of shape [batch_size, max_len], where 1 = valid, 0 = padding
    pub fn create_padding_mask_from_lengths(&self, lengths: &[usize], max_len: usize) -> Tensor {
        let batch_size = lengths.len();
        let mut mask_data = vec![0.0f32; batch_size * max_len];
        for (b, &len) in lengths.iter().enumerate() {
            for i in 0..len.min(max_len) {
                mask_data[b * max_len + i] = 1.0;
            }
        }
        let mask_arr = ArrayD::from_shape_vec(IxDyn(&[batch_size, max_len]), mask_data).unwrap();
        Tensor::new_literal(&self.context, mask_arr, "padding_mask")
    }

    // Legacy methods for backward compatibility

    /// Helper function to split tensor into heads.
    fn split_heads(&self, x: &Tensor) -> Tensor {
        x.reshape(vec![
            1,
            1,
            self.num_heads as i64,
            self.head_dim as i64,
        ])
        .transpose(1, 2)
    }

    /// Helper function to merge heads.
    fn combine_heads(&self, x: &Tensor) -> Tensor {
        x.transpose(1, 2).reshape(vec![
            1,
            self.embed_dim as i64,
        ])
    }
}

impl Module for MultiHeadAttention {
    /// Self-attention forward pass (Q = K = V).
    fn forward(&self, inputs: &Tensor) -> Tensor {
        let q = self.w_q.forward(inputs);
        let k = self.w_k.forward(inputs);
        let v = self.w_v.forward(inputs);

        let q_heads = self.split_heads(&q);
        let k_heads = self.split_heads(&k);
        let v_heads = self.split_heads(&v);

        let k_heads_transposed = k_heads.transpose(2, 3);
        let scores = q_heads.dot(&k_heads_transposed);

        let scale_tensor = Tensor::new_literal(
            &self.context,
            arr0(self.scale).into_dyn(),
            "scale",
        );
        let scores_scaled = &scores * &scale_tensor;

        let attention_weights = scores_scaled.softmax();
        let attention_output = attention_weights.dot(&v_heads);

        let concatenated_output = self.combine_heads(&attention_output);

        self.w_o.forward(&concatenated_output)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.w_q.parameters());
        params.extend(self.w_k.parameters());
        params.extend(self.w_v.parameters());
        params.extend(self.w_o.parameters());
        params
    }
}

/// Creates causal mask for decoder self-attention.
///
/// # Arguments
/// * `context` - Graph context
/// * `seq_len` - Sequence length
///
/// # Returns
/// Tensor of shape [1, 1, seq_len, seq_len] with 0 for valid positions and -1e9 for masked.
pub fn create_causal_mask(context: &Rc<RefCell<GraphContext>>, seq_len: usize) -> Tensor {
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j > i {
                mask_data[i * seq_len + j] = -1e9;
            }
        }
    }
    let mask_arr = ArrayD::from_shape_vec(IxDyn(&[1, 1, seq_len, seq_len]), mask_data).unwrap();
    Tensor::new_literal(context, mask_arr, "causal_mask")
}

/// Creates padding mask from key tensor.
///
/// # Arguments
/// * `context` - Graph context
/// * `padding_idx` - Padding token index
/// * `key_ids` - Tensor of token indices [batch, seq_len]
///
/// # Returns
/// Mask tensor [batch, 1, 1, seq_len] with 0 for valid and -1e9 for padding.
pub fn create_padding_mask_from_ids(
    context: &Rc<RefCell<GraphContext>>,
    lengths: &[usize],
    max_len: usize,
) -> Tensor {
    let batch_size = lengths.len();
    let mut mask_data = vec![-1e9f32; batch_size * max_len];
    for (b, &len) in lengths.iter().enumerate() {
        for i in 0..len.min(max_len) {
            mask_data[b * max_len + i] = 0.0;
        }
    }
    let mask_arr = ArrayD::from_shape_vec(IxDyn(&[batch_size, 1, 1, max_len]), mask_data).unwrap();
    Tensor::new_literal(context, mask_arr, "padding_mask")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mha_creation() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let mha = MultiHeadAttention::new(&context, 64, 4, "mha");

        assert_eq!(mha.embed_dim, 64);
        assert_eq!(mha.num_heads, 4);
        assert_eq!(mha.head_dim, 16);
    }

    #[test]
    fn test_causal_mask() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let mask = create_causal_mask(&context, 4);

        // Mask should be created
        // Check by getting the literal
        let graph = context.borrow();
        let main_graph = graph.main_graph();
        let node = main_graph.get_node(mask.node_id).unwrap();

        if let NodeType::Literal(Value::Tensor(arr)) = &node.node_type {
            assert_eq!(arr.shape(), &[1, 1, 4, 4]);
            // Check that upper triangle is -1e9
            assert!(arr[[0, 0, 0, 1]] < -1e8);
            assert!(arr[[0, 0, 0, 2]] < -1e8);
            assert!(arr[[0, 0, 0, 3]] < -1e8);
            // Diagonal and below = 0
            assert_eq!(arr[[0, 0, 0, 0]], 0.0);
            assert_eq!(arr[[0, 0, 1, 0]], 0.0);
            assert_eq!(arr[[0, 0, 1, 1]], 0.0);
        } else {
            panic!("Expected Literal tensor");
        }
    }

    #[test]
    fn test_padding_mask_from_lengths() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let mha = MultiHeadAttention::new(&context, 64, 4, "mha");
        let mask = mha.create_padding_mask_from_lengths(&[3, 2, 4], 4);

        let graph = context.borrow();
        let main_graph = graph.main_graph();
        let node = main_graph.get_node(mask.node_id).unwrap();

        if let NodeType::Literal(Value::Tensor(arr)) = &node.node_type {
            assert_eq!(arr.shape(), &[3, 4]);
            // Batch 0: length 3 -> [1, 1, 1, 0]
            assert_eq!(arr[[0, 0]], 1.0);
            assert_eq!(arr[[0, 2]], 1.0);
            assert_eq!(arr[[0, 3]], 0.0);
            // Batch 1: length 2 -> [1, 1, 0, 0]
            assert_eq!(arr[[1, 1]], 1.0);
            assert_eq!(arr[[1, 2]], 0.0);
        } else {
            panic!("Expected Literal tensor");
        }
    }

    #[test]
    fn test_mha_parameters() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let mha = MultiHeadAttention::new(&context, 64, 4, "mha");
        let params = mha.parameters();

        // Should have 8 parameters: weight + bias for each of w_q, w_k, w_v, w_o
        assert_eq!(params.len(), 8);
    }

    #[test]
    fn test_mha_config() {
        let config = MultiHeadAttentionConfig::new(256, 4)
            .with_dropout(0.1)
            .without_bias();

        assert_eq!(config.embed_dim, 256);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.dropout, 0.1);
        assert!(!config.bias);
    }
}
