//  src/nn/transformer.rs  (финальная версия — без заглушек)
//! Один блок кодировщика Трансформера в графовой архитектуре.

use crate::nn::{FeedForward, LayerNorm, Module, MultiHeadAttention};
use crate::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Один блок кодировщика Трансформера.
pub struct TransformerBlock {
    attention: MultiHeadAttention,
    norm1: LayerNorm,
    feed_forward: FeedForward,
    norm2: LayerNorm,
}

impl TransformerBlock {
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        embed_dim: usize,
        num_heads: usize,
        _ff_hidden_dim: usize, // оставлено для совместимости вызова
        name: &str,
    ) -> Self {
        let attn_name = format!("{}.mha", name);
        let ff_name = format!("{}.ff", name);
        let norm1_name = format!("{}.norm1", name);
        let norm2_name = format!("{}.norm2", name);

        Self {
            attention: MultiHeadAttention::new(context, embed_dim, num_heads, &attn_name),
            norm1: LayerNorm::new(context, &norm1_name),
            feed_forward: FeedForward::new(context, &ff_name),
            norm2: LayerNorm::new(context, &norm2_name),
        }
    }
}

impl Module for TransformerBlock {
    /// Полный forward с остаточными связями и двумя LayerNorm.
    fn forward(&self, inputs: &Tensor) -> Tensor {
        let normed1 = self.norm1.forward(inputs);
        let attn_out = self.attention.forward(&normed1);
        let x = inputs + &attn_out;

        let normed2 = self.norm2.forward(&x);
        let ff_out = self.feed_forward.forward(&normed2);
        let final_out = &x + &ff_out;

        final_out
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