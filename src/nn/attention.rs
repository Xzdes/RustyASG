//! Модуль, реализующий многоголовое внимание (Multi-Head Self-Attention)
//! для графовой архитектуры.

use crate::nn::{Linear, Module};
use crate::tensor::{GraphContext, Tensor};
use ndarray::arr0;
use std::cell::RefCell;
use std::rc::Rc;

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
    /// Контекст графа, необходимый для создания констант.
    context: Rc<RefCell<GraphContext>>,
}

impl MultiHeadAttention {
    /// Создает новый слой MultiHeadAttention.
    ///
    /// # Аргументы
    ///
    /// * `context` - Ссылка на `GraphContext`.
    /// * `embed_dim` - Размерность встраиваний (входных векторов).
    /// * `num_heads` - Количество "голов" внимания. Должно делить `embed_dim` без остатка.
    /// * `name` - Уникальное имя для слоя.
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
            w_q: Linear::new(context, embed_dim, embed_dim, &w_q_name),
            w_k: Linear::new(context, embed_dim, embed_dim, &w_k_name),
            w_v: Linear::new(context, embed_dim, embed_dim, &w_v_name),
            w_o: Linear::new(context, embed_dim, embed_dim, &w_o_name),
            context: Rc::clone(context),
        }
    }

    /// Вспомогательная функция для разделения тензора на головы.
    /// Исходная форма: [batch_size, seq_len, embed_dim]
    /// Выходная форма: [batch_size, num_heads, seq_len, head_dim]
    fn split_heads(&self, x: &Tensor, batch_size: i64, seq_len: i64) -> Tensor {
        x.reshape(vec![
            batch_size,
            seq_len,
            self.num_heads as i64,
            self.head_dim as i64,
        ])
        .transpose(1, 2)
    }

    /// Вспомогательная функция для слияния голов.
    /// Исходная форма: [batch_size, num_heads, seq_len, head_dim]
    /// Выходная форма: [batch_size, seq_len, embed_dim]
    fn combine_heads(&self, x: &Tensor, batch_size: i64, seq_len: i64) -> Tensor {
        x.transpose(1, 2).reshape(vec![
            batch_size,
            seq_len,
            self.embed_dim as i64,
        ])
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        // В графовой модели мы не знаем конкретных размеров батча и последовательности.
        // Для нашего примера с batch_size=1 и seq_len=1 это будет работать.
        // В более продвинутой системе здесь нужны были бы символьные операции для
        // получения размеров.
        let batch_size = 1;
        let seq_len = 1;

        // 1. Линейные проекции для Q, K, V
        let q = self.w_q.forward(inputs);
        let k = self.w_k.forward(inputs);
        let v = self.w_v.forward(inputs);

        // 2. Разделение на "головы"
        let q_heads = self.split_heads(&q, batch_size, seq_len);
        let k_heads = self.split_heads(&k, batch_size, seq_len);
        let v_heads = self.split_heads(&v, batch_size, seq_len);

        // 3. Вычисление очков внимания: scores = (Q * K^T) / sqrt(d_k)
        let k_heads_transposed = k_heads.transpose(2, 3);
        let scores = q_heads.dot(&k_heads_transposed);

        // Масштабирование
        let scale_factor = (self.head_dim as f32).sqrt();
        let scale_tensor = Tensor::new_literal(
            &self.context,
            arr0(1.0 / scale_factor).into_dyn(),
            "scale",
        );
        let scores_scaled = &scores * &scale_tensor;

        // Применение Softmax
        let attention_weights = scores_scaled.softmax();

        // 4. Взвешивание векторов значений (V)
        let attention_output = attention_weights.dot(&v_heads);

        // 5. Слияние голов
        let concatenated_output = self.combine_heads(&attention_output, batch_size, seq_len);
            
        // 6. Выходная линейная проекция
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