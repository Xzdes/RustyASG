//! Модуль, реализующий один блок кодировщика Трансформера для графовой архитектуры.

use crate::nn::{FeedForward, LayerNorm, Module, MultiHeadAttention};
use crate::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Один блок кодировщика Трансформера.
pub struct TransformerBlock {
    /// Слой многоголового внимания.
    attention: MultiHeadAttention,
    /// Слой нормализации перед attention.
    norm1: LayerNorm,
    /// Полносвязная сеть.
    feed_forward: FeedForward,
    /// Слой нормализации перед feed-forward.
    norm2: LayerNorm,
}

impl TransformerBlock {
    /// Создает новый блок Трансформера.
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
    /// Прямой проход через блок Трансформера, строящий подграф операций.
    fn forward(&self, inputs: &Tensor) -> Tensor {
        // --- ВРЕМЕННОЕ УПРОЩЕНИЕ ДЛЯ ОТЛАДКИ ---
        // Мы убираем LayerNorm и остаточные соединения, чтобы проверить основной путь градиента.

        // 1. Под-слой Multi-Head Attention
        let attention_output = self.attention.forward(inputs);
        
        // 2. Под-слой FeedForward
        let final_output = self.feed_forward.forward(&attention_output);

        final_output
    }

    /// Собирает параметры из всех вложенных модулей.
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.attention.parameters());
        // params.extend(self.norm1.parameters()); // Временно отключаем
        params.extend(self.feed_forward.parameters());
        // params.extend(self.norm2.parameters()); // Временно отключаем
        params
    }
}