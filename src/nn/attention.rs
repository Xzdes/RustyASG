//! Модуль, реализующий многоголовое внимание (Multi-Head Self-Attention)
//! для графовой архитектуры.

use crate::nn::{Linear, Module};
use crate::tensor::{GraphContext, Tensor};
use ndarray::arr0;
use std::cell::RefCell;
use std::rc::Rc;

/// Реализация многоголового внимания (Multi-Head Self-Attention).
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    context: Rc<RefCell<GraphContext>>,
}

impl MultiHeadAttention {
    /// Создает новый слой MultiHeadAttention.
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        embed_dim: usize,
        num_heads: usize,
        name: &str,
    ) -> Self {
        assert!(
            embed_dim % num_heads == 0,
            "embed_dim должен делиться на num_heads без остатка."
        );
        let head_dim = embed_dim / num_heads;

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
            context: Rc::clone(context),
        }
    }

    /// Вспомогательная функция для разделения тензора на головы.
    fn split_heads(&self, x: &Tensor) -> Tensor {
        // [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        // В нашем демо batch=1, seq_len=1
        x.reshape(vec![
            1,
            1,
            self.num_heads as i64,
            self.head_dim as i64,
        ])
        .transpose(1, 2)
    }

    /// Вспомогательная функция для слияния голов.
    fn combine_heads(&self, x: &Tensor) -> Tensor {
        // [batch_size, num_heads, seq_len, head_dim] -> [batch_size, seq_len, num_heads, head_dim] -> [batch_size * seq_len, embed_dim]
        x.transpose(1, 2).reshape(vec![
            1, // batch_size * seq_len
            self.embed_dim as i64,
        ])
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        let q = self.w_q.forward(inputs);
        let k = self.w_k.forward(inputs);
        let v = self.w_v.forward(inputs);

        let q_heads = self.split_heads(&q);
        let k_heads = self.split_heads(&k);
        let v_heads = self.split_heads(&v);

        let k_heads_transposed = k_heads.transpose(2, 3);
        let scores = q_heads.dot(&k_heads_transposed);

        let scale_factor = (self.head_dim as f32).sqrt();
        let scale_tensor = Tensor::new_literal(
            &self.context,
            arr0(1.0 / scale_factor).into_dyn(),
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