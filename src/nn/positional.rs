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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asg::{NodeType, Value};
    use crate::runtime::{backend::Backend, cpu_backend::CpuBackend};
    use std::collections::HashMap;

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
