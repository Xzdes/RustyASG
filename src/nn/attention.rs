//! Модуль, реализующий многоголовое внимание (Multi-Head Self-Attention).

use crate::nn::{Linear, Module};
use crate::tensor::Tensor;
use crate::error::Result;
use std::ops::Mul;

/// Реализация многоголового внимания (Multi-Head Self-Attention).
///
/// Этот слой позволяет модели одновременно обращать внимание на информацию
/// из разных подпространств представлений в разных позициях. Это ключевой
/// компонент архитектуры Трансформер.
pub struct MultiHeadAttention {
    /// Количество "голов" внимания.
    num_heads: usize,
    /// Размерность каждой "головы".
    head_dim: usize,
    /// Общая размерность встраиваний.
    embed_dim: usize,
    /// Линейный слой для проекции Query.
    w_q: Linear,
    /// Линейный слой для проекции Key.
    w_k: Linear,
    /// Линейный слой для проекции Value.
    w_v: Linear,
    /// Выходной линейный слой.
    w_o: Linear,
}

impl MultiHeadAttention {
    /// Создает новый слой MultiHeadAttention.
    ///
    /// # Аргументы
    /// * `embed_dim` - Размерность встраиваний (входных векторов).
    /// * `num_heads` - Количество "голов" внимания. Должно делить `embed_dim` без остатка.
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        // Эта проверка критична для корректной работы.
        // В будущем ее можно будет заменить на возвращение ошибки `Result::Err`.
        assert!(
            embed_dim % num_heads == 0,
            "embed_dim должен делиться на num_heads без остатка."
        );

        let head_dim = embed_dim / num_heads;
        Self {
            num_heads,
            head_dim,
            embed_dim,
            w_q: Linear::new(embed_dim, embed_dim),
            w_k: Linear::new(embed_dim, embed_dim),
            w_v: Linear::new(embed_dim, embed_dim),
            w_o: Linear::new(embed_dim, embed_dim),
        }
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        let batch_size = inputs.data.borrow().shape()[0];
        let seq_len = inputs.data.borrow().shape()[1];

        // 1. Линейные проекции для Q, K, V
        let q = self.w_q.forward(inputs)?;
        let k = self.w_k.forward(inputs)?;
        let v = self.w_v.forward(inputs)?;

        // 2. Разделение на "головы" и транспонирование
        let q_heads = q
            .reshape(vec![batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let k_heads = k
            .reshape(vec![batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let v_heads = v
            .reshape(vec![batch_size, seq_len, self.num_heads, self.head_dim])?
            .transpose(1, 2)?;

        // 3. Вычисление очков внимания
        let k_heads_transposed = k_heads.transpose(2, 3)?;
        // --- ИЗМЕНЕНИЕ: Добавляем `?` после `.dot()` ---
        let scores = q_heads.dot(&k_heads_transposed)?;

        // Масштабирование
        let scale_inv = Tensor::new(
            ndarray::arr0(1.0 / (self.head_dim as f32).sqrt()).into_dyn(),
            false,
        );
        let scores_scaled = scores.mul(&scale_inv);

        // Применение Softmax
        let attention_weights = scores_scaled.softmax();

        // 4. Взвешивание векторов значений (V)
        // --- ИЗМЕНЕНИЕ: Добавляем `?` после `.dot()` ---
        let attention_output = attention_weights.dot(&v_heads)?;

        // 5. Слияние голов
        let concatenated_output = attention_output
            .transpose(1, 2)?
            .reshape(vec![batch_size, seq_len, self.embed_dim])?;
            
        // 6. Выходная линейная проекция
        let final_output = self.w_o.forward(&concatenated_output)?;
            
        Ok(final_output)
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