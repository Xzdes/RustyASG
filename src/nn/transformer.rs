//! Модуль, реализующий один блок кодировщика Трансформера для графовой архитектуры.

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
        ff_hidden_dim: usize,
        name: &str,
    ) -> Self {
        let attn_name = format!("{}.mha", name);
        let ff_name = format!("{}.ff", name);
        let norm1_name = format!("{}.norm1", name);
        let norm2_name = format!("{}.norm2", name);

        Self {
            attention: MultiHeadAttention::new(context, embed_dim, num_heads, &attn_name),
            norm1: LayerNorm::new(context, &norm1_name),
            feed_forward: FeedForward::new(context, embed_dim, ff_hidden_dim, &ff_name),
            norm2: LayerNorm::new(context, &norm2_name),
        }
    }
}

impl Module for TransformerBlock {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        // Возвращаем полную архитектуру с LayerNorm и остаточными соединениями
        let normed_inputs1 = self.norm1.forward(inputs);
        let attention_output = self.attention.forward(&normed_inputs1);
        let x = inputs + &attention_output;

        let normed_inputs2 = self.norm2.forward(&x);
        let ff_output = self.feed_forward.forward(&normed_inputs2);
        let final_output = &x + &ff_output;

        final_output
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