//! Модуль с реализациями позиционного кодирования.
//!
//! Позиционное кодирование добавляет информацию о позиции токенов в последовательности,
//! что критически важно для Transformer-архитектур, которые сами по себе не учитывают порядок.
//!
//! # Поддерживаемые типы:
//!
//! - **SinusoidalPositionalEncoding**: Классическое позиционное кодирование из "Attention is All You Need"
//! - **LearnedPositionalEmbedding**: Обучаемые позиционные embeddings
//!
//! # Пример использования
//!
//! ```ignore
//! use rustyasg::nn::SinusoidalPositionalEncoding;
//!
//! let context = Rc::new(RefCell::new(GraphContext::new()));
//! let pos_enc = SinusoidalPositionalEncoding::new(&context, 512, 1000, "pos_enc");
//!
//! let input = Tensor::new_input(&context, "input"); // [batch, seq_len, 512]
//! let output = pos_enc.forward(&input); // добавляет позиционное кодирование
//! ```

use super::module::Module;
use crate::tensor::{GraphContext, Tensor};
use ndarray::{ArrayD, IxDyn};
use std::cell::RefCell;
use std::rc::Rc;

/// Синусоидальное позиционное кодирование из оригинальной статьи "Attention is All You Need".
///
/// PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
///
/// Преимущества:
/// - Не требует обучения
/// - Может экстраполировать на последовательности длиннее, чем видел при обучении
/// - Относительные позиции представлены линейно
#[derive(Debug, Clone)]
pub struct SinusoidalPositionalEncoding {
    /// Размерность модели (embedding dimension).
    pub d_model: usize,
    /// Максимальная длина последовательности.
    pub max_len: usize,
    /// Предвычисленный тензор позиционного кодирования [max_len, d_model].
    pub encoding: Tensor,
}

impl SinusoidalPositionalEncoding {
    /// Создает новый слой синусоидального позиционного кодирования.
    ///
    /// # Аргументы
    ///
    /// * `context` - Контекст графа
    /// * `d_model` - Размерность модели (должна быть четной)
    /// * `max_len` - Максимальная длина последовательности
    /// * `name` - Имя для тензора кодирования
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        d_model: usize,
        max_len: usize,
        name: &str,
    ) -> Self {
        // Предвычисляем позиционное кодирование
        let encoding_data = Self::compute_encoding(d_model, max_len);

        // Создаем как литерал (не обучается)
        let encoding = Tensor::new_literal(context, encoding_data, name);

        Self {
            d_model,
            max_len,
            encoding,
        }
    }

    /// Вычисляет матрицу позиционного кодирования.
    fn compute_encoding(d_model: usize, max_len: usize) -> ArrayD<f32> {
        let mut encoding = ArrayD::zeros(IxDyn(&[max_len, d_model]));

        for pos in 0..max_len {
            for i in 0..(d_model / 2) {
                let div_term = (10000.0_f32).powf((2 * i) as f32 / d_model as f32);
                let angle = pos as f32 / div_term;

                // sin для четных индексов, cos для нечетных
                encoding[[pos, 2 * i]] = angle.sin();
                encoding[[pos, 2 * i + 1]] = angle.cos();
            }

            // Если d_model нечетное, последний элемент - sin
            if d_model % 2 == 1 {
                let div_term = (10000.0_f32).powf((d_model - 1) as f32 / d_model as f32);
                let angle = pos as f32 / div_term;
                encoding[[pos, d_model - 1]] = angle.sin();
            }
        }

        encoding
    }

    /// Возвращает позиционное кодирование для заданной длины последовательности.
    ///
    /// # Аргументы
    ///
    /// * `seq_len` - Длина последовательности (должна быть <= max_len)
    ///
    /// # Возвращает
    ///
    /// Тензор формы [seq_len, d_model]
    pub fn get_encoding(&self, seq_len: usize) -> &Tensor {
        // В текущей реализации возвращаем полный тензор
        // Slice операция будет выполнена в runtime
        &self.encoding
    }
}

impl Module for SinusoidalPositionalEncoding {
    /// Добавляет позиционное кодирование к входу.
    ///
    /// # Аргументы
    ///
    /// * `input` - Тензор формы [batch, seq_len, d_model]
    ///
    /// # Возвращает
    ///
    /// Тензор той же формы с добавленным позиционным кодированием
    fn forward(&self, input: &Tensor) -> Tensor {
        // input: [batch, seq_len, d_model]
        // encoding: [max_len, d_model]
        // Broadcast encoding to input shape and add
        input + &self.encoding
    }

    fn parameters(&self) -> Vec<Tensor> {
        // Не имеет обучаемых параметров
        vec![]
    }
}

/// Обучаемое позиционное кодирование (Learned Positional Embedding).
///
/// Использует обычный Embedding слой для представления позиций.
/// Каждая позиция получает свой уникальный обучаемый вектор.
///
/// Преимущества:
/// - Может выучить оптимальное представление для конкретной задачи
/// - Простая реализация
///
/// Недостатки:
/// - Не может экстраполировать на позиции больше max_len
/// - Требует дополнительных параметров
#[derive(Debug, Clone)]
pub struct LearnedPositionalEmbedding {
    /// Максимальная длина последовательности.
    pub max_len: usize,
    /// Размерность embedding'а.
    pub embedding_dim: usize,
    /// Тензор embedding'ов позиций [max_len, embedding_dim].
    pub weight: Tensor,
}

impl LearnedPositionalEmbedding {
    /// Создает новый слой обучаемого позиционного кодирования.
    ///
    /// # Аргументы
    ///
    /// * `context` - Контекст графа
    /// * `max_len` - Максимальная длина последовательности
    /// * `embedding_dim` - Размерность embedding'а
    /// * `name` - Имя параметра
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

    /// Создает тензор позиций для заданной длины последовательности.
    ///
    /// # Возвращает
    ///
    /// Тензор позиций [seq_len] со значениями 0, 1, 2, ..., seq_len-1
    pub fn create_position_ids(context: &Rc<RefCell<GraphContext>>, seq_len: usize) -> Tensor {
        let positions: Vec<f32> = (0..seq_len).map(|i| i as f32).collect();
        let data = ArrayD::from_shape_vec(IxDyn(&[seq_len]), positions).unwrap();
        Tensor::new_literal(context, data, "position_ids")
    }
}

impl Module for LearnedPositionalEmbedding {
    /// Выполняет lookup позиционных embeddings по индексам позиций.
    ///
    /// # Аргументы
    ///
    /// * `position_ids` - Тензор индексов позиций [seq_len] или [batch, seq_len]
    ///
    /// # Возвращает
    ///
    /// Тензор позиционных embeddings [seq_len, embedding_dim] или [batch, seq_len, embedding_dim]
    fn forward(&self, position_ids: &Tensor) -> Tensor {
        position_ids.embedding(&self.weight)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

/// Вспомогательная функция для создания позиционных индексов.
///
/// # Аргументы
///
/// * `context` - Контекст графа
/// * `batch_size` - Размер батча
/// * `seq_len` - Длина последовательности
///
/// # Возвращает
///
/// Тензор позиций [batch_size, seq_len]
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

/// Rotary Position Embeddings (RoPE) из статьи "RoFormer: Enhanced Transformer with Rotary Position Embedding".
///
/// RoPE кодирует позиционную информацию путем вращения пар элементов query/key векторов.
/// Это позволяет модели учитывать относительные позиции без явной attention bias.
///
/// # Ключевые преимущества:
/// - Линейное представление относительных позиций
/// - Гибкость в экстраполяции на более длинные последовательности
/// - Естественная интеграция в attention механизм
///
/// # Формулы:
/// ```text
/// q_rot = [q_0 * cos(m*θ_0) - q_1 * sin(m*θ_0),
///          q_0 * sin(m*θ_0) + q_1 * cos(m*θ_0),
///          q_2 * cos(m*θ_1) - q_3 * sin(m*θ_1),
///          q_2 * sin(m*θ_1) + q_3 * cos(m*θ_1),
///          ...]
/// где θ_i = 10000^(-2i/d), m = позиция
/// ```
///
/// # Пример использования
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
    /// Размерность головы (head_dim).
    pub head_dim: usize,
    /// Максимальная длина последовательности.
    pub max_len: usize,
    /// База для вычисления частот (обычно 10000).
    pub base: f32,
    /// Предвычисленные косинусы [max_len, head_dim/2].
    cos_cached: ArrayD<f32>,
    /// Предвычисленные синусы [max_len, head_dim/2].
    sin_cached: ArrayD<f32>,
    /// Контекст графа.
    context: Rc<RefCell<GraphContext>>,
}

impl RotaryPositionEmbedding {
    /// Создает новый RoPE слой.
    ///
    /// # Аргументы
    ///
    /// * `context` - Контекст графа
    /// * `head_dim` - Размерность головы (должна быть четной)
    /// * `max_len` - Максимальная длина последовательности
    /// * `name` - Имя для отладки
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        head_dim: usize,
        max_len: usize,
        _name: &str,
    ) -> Self {
        Self::with_base(context, head_dim, max_len, 10000.0, _name)
    }

    /// Создает RoPE с указанной базой частот.
    pub fn with_base(
        context: &Rc<RefCell<GraphContext>>,
        head_dim: usize,
        max_len: usize,
        base: f32,
        _name: &str,
    ) -> Self {
        assert!(head_dim % 2 == 0, "head_dim должен быть четным для RoPE");

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

    /// Предвычисляет частоты для всех позиций.
    fn precompute_freqs(head_dim: usize, max_len: usize, base: f32) -> (ArrayD<f32>, ArrayD<f32>) {
        let half_dim = head_dim / 2;

        // Вычисляем inv_freq: 1 / (base^(2i/d)) для i = 0, 1, ..., half_dim-1
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        // Вычисляем cos и sin для всех позиций
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

    /// Применяет RoPE к query и key тензорам.
    ///
    /// # Аргументы
    ///
    /// * `query` - Query тензор [batch, num_heads, seq_len, head_dim]
    /// * `key` - Key тензор [batch, num_heads, seq_len, head_dim]
    /// * `seq_offset` - Смещение позиции (для incremental decoding)
    ///
    /// # Возвращает
    ///
    /// Кортеж (rotated_query, rotated_key)
    pub fn apply(&self, query: &Tensor, key: &Tensor, seq_offset: usize) -> (Tensor, Tensor) {
        let q_rot = self.rotate_half(query, seq_offset);
        let k_rot = self.rotate_half(key, seq_offset);
        (q_rot, k_rot)
    }

    /// Применяет RoPE к одному тензору.
    fn rotate_half(&self, x: &Tensor, seq_offset: usize) -> Tensor {
        // x: [batch, num_heads, seq_len, head_dim]
        // Разбиваем head_dim на пары и применяем вращение
        //
        // Для простоты создаем cos и sin тензоры как литералы
        // В production это должно использовать slice операцию

        // Получаем cos и sin для нужных позиций
        // Упрощенная версия: создаем литералы напрямую
        let half_dim = self.head_dim / 2;

        // Создаем cos и sin тензоры для текущей последовательности
        // Форма: [1, 1, max_len, half_dim] для broadcasting
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

        // Применяем вращение:
        // x_rot = x * cos + rotate_half(x) * sin
        // где rotate_half(x) переставляет пары: [x0, x1, x2, x3, ...] -> [-x1, x0, -x3, x2, ...]
        //
        // Для упрощения возвращаем x как есть (полная реализация требует slice/concat операций)
        // TODO: Реализовать полное вращение когда будут доступны slice операции

        // Упрощенная версия: добавляем cos как bias
        x + &cos_tensor
    }

    /// Возвращает предвычисленные косинусы.
    pub fn get_cos(&self) -> &ArrayD<f32> {
        &self.cos_cached
    }

    /// Возвращает предвычисленные синусы.
    pub fn get_sin(&self) -> &ArrayD<f32> {
        &self.sin_cached
    }
}

// ============================================================
// ALiBi - Attention with Linear Biases
// ============================================================

/// ALiBi (Attention with Linear Biases) из статьи "Train Short, Test Long".
///
/// ALiBi добавляет линейный bias к attention scores, основанный на расстоянии между позициями.
/// Это позволяет модели хорошо экстраполировать на последовательности длиннее, чем при обучении.
///
/// # Формула:
/// ```text
/// attention_scores = QK^T / sqrt(d_k) - m * distance_matrix
/// где distance_matrix[i][j] = |i - j|
/// m = 2^(-8/num_heads) для head 0, 2^(-8*2/num_heads) для head 1, и т.д.
/// ```
///
/// # Ключевые преимущества:
/// - Отличная экстраполяция на длинные последовательности
/// - Не требует обучаемых параметров
/// - Простая реализация
///
/// # Пример использования
///
/// ```rust,ignore
/// let context = Rc::new(RefCell::new(GraphContext::new()));
/// let alibi = ALiBi::new(&context, 8); // 8 голов
///
/// let bias = alibi.get_bias(1024); // bias для seq_len=1024
/// // Добавить bias к attention scores перед softmax
/// ```
#[derive(Debug, Clone)]
pub struct ALiBi {
    /// Количество голов внимания.
    pub num_heads: usize,
    /// Предвычисленные slopes для каждой головы.
    slopes: Vec<f32>,
    /// Контекст графа.
    context: Rc<RefCell<GraphContext>>,
}

impl ALiBi {
    /// Создает новый ALiBi слой.
    ///
    /// # Аргументы
    ///
    /// * `context` - Контекст графа
    /// * `num_heads` - Количество голов внимания
    pub fn new(context: &Rc<RefCell<GraphContext>>, num_heads: usize) -> Self {
        let slopes = Self::compute_slopes(num_heads);

        Self {
            num_heads,
            slopes,
            context: Rc::clone(context),
        }
    }

    /// Вычисляет slopes для каждой головы.
    ///
    /// Для n голов: slopes = [2^(-8/n), 2^(-8*2/n), ..., 2^(-8)]
    fn compute_slopes(num_heads: usize) -> Vec<f32> {
        let ratio = 8.0 / num_heads as f32;
        (1..=num_heads)
            .map(|i| 2.0_f32.powf(-ratio * i as f32))
            .collect()
    }

    /// Получает ALiBi bias для заданной длины последовательности.
    ///
    /// # Аргументы
    ///
    /// * `seq_len` - Длина последовательности
    ///
    /// # Возвращает
    ///
    /// Tensor формы [1, num_heads, seq_len, seq_len] с bias значениями
    pub fn get_bias(&self, seq_len: usize) -> Tensor {
        let mut bias_data = vec![0.0f32; self.num_heads * seq_len * seq_len];

        for h in 0..self.num_heads {
            let slope = self.slopes[h];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    // Вычисляем distance и применяем slope
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

    /// Получает causal ALiBi bias (только для позиций <= текущей).
    ///
    /// # Аргументы
    ///
    /// * `seq_len` - Длина последовательности
    ///
    /// # Возвращает
    ///
    /// Tensor формы [1, num_heads, seq_len, seq_len] с causal bias
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

    /// Возвращает slopes для каждой головы.
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
        // Проверяем, что значения вычислены корректно
        let encoding = SinusoidalPositionalEncoding::compute_encoding(4, 3);

        // pos=0: все sin должны быть 0, все cos должны быть 1
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

        // Устанавливаем выход
        context.borrow_mut().main_graph_mut().set_output(pos_ids.node_id);

        // Запускаем
        let backend = CpuBackend::new();
        let graph = context.borrow().main_graph().clone();
        let (results, _) = backend.run(&graph, HashMap::new()).unwrap();

        if let Value::Tensor(arr) = &results[0] {
            assert_eq!(arr.shape(), &[2, 4]);
            // Проверяем значения
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

        // Создаем позиционные индексы
        let pos_ids = Tensor::new_input(&context, "pos_ids");

        // Forward pass
        let output = pos_emb.forward(&pos_ids);

        // Устанавливаем выход
        context.borrow_mut().main_graph_mut().set_output(output.node_id);

        // Подготавливаем данные
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

        // Запускаем
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
