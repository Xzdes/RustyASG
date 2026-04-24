//! A single Transformer encoder block for the graph-based architecture.
//!
//! Consists of multi-head self-attention, feed-forward network, and two
//! layer-norms with residual connections. Uses the "Pre-LN" variant
//! (LayerNorm before the sublayer, residual added after) which is more stable
//! for deep models.

use crate::nn::{FeedForward, LayerNorm, Module, MultiHeadAttention};
use crate::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// One Transformer encoder block.
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    norm1: LayerNorm,
    feed_forward: FeedForward,
    norm2: LayerNorm,
}

impl TransformerBlock {
    /// Creates a Transformer block and registers all its parameter shapes.
    ///
    /// # Arguments
    /// * `context` — shared graph context.
    /// * `embed_dim` — model dimensionality.
    /// * `num_heads` — number of attention heads (must divide `embed_dim`).
    /// * `ff_hidden_dim` — FeedForward internal size (typically `4 * embed_dim`).
    /// * `name` — parameter-name prefix.
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        embed_dim: usize,
        num_heads: usize,
        ff_hidden_dim: usize,
        name: &str,
    ) -> Self {
        Self {
            attention: MultiHeadAttention::new(
                context,
                embed_dim,
                num_heads,
                &format!("{}.mha", name),
            ),
            norm1: LayerNorm::new(context, &format!("{}.norm1", name), embed_dim),
            feed_forward: FeedForward::new(
                context,
                &format!("{}.ff", name),
                embed_dim,
                ff_hidden_dim,
            ),
            norm2: LayerNorm::new(context, &format!("{}.norm2", name), embed_dim),
        }
    }
}

impl Module for TransformerBlock {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        let normed1 = self.norm1.forward(inputs);
        let attn_out = self.attention.forward(&normed1);
        let x = inputs + &attn_out;

        let normed2 = self.norm2.forward(&x);
        let ff_out = self.feed_forward.forward(&normed2);
        &x + &ff_out
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.attention.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.feed_forward.parameters());
        params.extend(self.norm2.parameters());
        params
    }
}
