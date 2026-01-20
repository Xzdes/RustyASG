//! Модуль с реализацией слоя Embedding.
//!
//! Embedding слой преобразует целочисленные индексы в плотные векторы
//! фиксированной размерности. Широко используется в NLP для представления
//! слов, токенов и других дискретных сущностей.

use super::module::Module;
use crate::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Слой Embedding для преобразования индексов в плотные векторы.
///
/// # Пример использования
///
/// ```ignore
/// use rustyasg::nn::Embedding;
///
/// let context = Rc::new(RefCell::new(GraphContext::new()));
/// let embedding = Embedding::new(&context, 10000, 256, "token_embedding");
/// // vocab_size=10000, embedding_dim=256
///
/// let indices = Tensor::new_input(&context, "token_ids"); // [batch, seq_len]
/// let embedded = embedding.forward(&indices); // [batch, seq_len, 256]
/// ```
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Количество уникальных индексов (размер словаря).
    pub num_embeddings: usize,
    /// Размерность embedding-вектора.
    pub embedding_dim: usize,
    /// Матрица embedding'ов формы [num_embeddings, embedding_dim].
    pub weight: Tensor,
}

impl Embedding {
    /// Создает новый слой Embedding.
    ///
    /// # Аргументы
    ///
    /// * `context` - Контекст графа
    /// * `num_embeddings` - Размер словаря (количество уникальных индексов)
    /// * `embedding_dim` - Размерность выходных векторов
    /// * `name` - Имя для параметров (используется как префикс)
    ///
    /// # Возвращает
    ///
    /// Новый экземпляр `Embedding` с весами формы [num_embeddings, embedding_dim]
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        num_embeddings: usize,
        embedding_dim: usize,
        name: &str,
    ) -> Self {
        let weight = Tensor::new_parameter(context, &format!("{}_weight", name));

        Self {
            num_embeddings,
            embedding_dim,
            weight,
        }
    }

    /// Создает слой Embedding с предоставленным тензором весов.
    ///
    /// Полезно для загрузки предобученных embedding'ов.
    pub fn from_weight(weight: Tensor, num_embeddings: usize, embedding_dim: usize) -> Self {
        Self {
            num_embeddings,
            embedding_dim,
            weight,
        }
    }
}

impl Module for Embedding {
    /// Выполняет embedding lookup.
    ///
    /// # Аргументы
    ///
    /// * `indices` - Тензор индексов любой формы [*]
    ///
    /// # Возвращает
    ///
    /// Тензор формы [*, embedding_dim]
    fn forward(&self, input: &Tensor) -> Tensor {
        input.embedding(&self.weight)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asg::Value;
    use crate::runtime::{backend::Backend, cpu_backend::CpuBackend};
    use ndarray::{arr2, ArrayD, IxDyn};
    use std::collections::HashMap;

    #[test]
    fn test_embedding_creation() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let embedding = Embedding::new(&context, 1000, 64, "emb");

        assert_eq!(embedding.num_embeddings, 1000);
        assert_eq!(embedding.embedding_dim, 64);
        assert_eq!(embedding.parameters().len(), 1);
    }

    #[test]
    fn test_embedding_forward() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let embedding = Embedding::new(&context, 5, 3, "emb");

        // Создаем входные индексы
        let indices = Tensor::new_input(&context, "indices");

        // Forward pass
        let output = embedding.forward(&indices);

        // Устанавливаем выход графа
        context.borrow_mut().main_graph_mut().set_output(output.node_id);

        // Подготавливаем данные
        // Embedding weights: 5x3 матрица
        let weight_data = arr2(&[
            [1.0, 2.0, 3.0],    // index 0
            [4.0, 5.0, 6.0],    // index 1
            [7.0, 8.0, 9.0],    // index 2
            [10.0, 11.0, 12.0], // index 3
            [13.0, 14.0, 15.0], // index 4
        ]).into_dyn();

        // Indices: [2, 3] -> выбираем index 0 и index 2
        let indices_data = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 2.0]).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("indices".to_string(), Value::Tensor(indices_data));
        inputs.insert("emb_weight".to_string(), Value::Tensor(weight_data));

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
                        crate::asg::NodeType::Input { name: n } |
                        crate::asg::NodeType::Parameter { name: n } if n == &name
                    )
                })
                .map(|(id, _)| *id);

            if let Some(id) = node_id {
                memo.insert((0, id), value);
            }
        }

        let graph = context.borrow().main_graph().clone();
        let (results, _) = backend.run(&graph, memo).unwrap();

        // Проверяем результат
        let result = &results[0];
        if let Value::Tensor(arr) = result {
            assert_eq!(arr.shape(), &[2, 3]); // [num_indices, embedding_dim]

            // index 0 -> [1, 2, 3]
            assert!((arr[[0, 0]] - 1.0).abs() < 1e-5);
            assert!((arr[[0, 1]] - 2.0).abs() < 1e-5);
            assert!((arr[[0, 2]] - 3.0).abs() < 1e-5);

            // index 2 -> [7, 8, 9]
            assert!((arr[[1, 0]] - 7.0).abs() < 1e-5);
            assert!((arr[[1, 1]] - 8.0).abs() < 1e-5);
            assert!((arr[[1, 2]] - 9.0).abs() < 1e-5);
        } else {
            panic!("Expected tensor output");
        }
    }

    #[test]
    fn test_embedding_2d_indices() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let embedding = Embedding::new(&context, 5, 4, "emb");

        let indices = Tensor::new_input(&context, "indices");
        let output = embedding.forward(&indices);

        context.borrow_mut().main_graph_mut().set_output(output.node_id);

        // Embedding weights: 5x4
        let weight_data = ArrayD::from_shape_vec(
            IxDyn(&[5, 4]),
            (0..20).map(|x| x as f32).collect()
        ).unwrap();

        // Indices: [2, 3] (batch=2, seq_len=3)
        let indices_data = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 0.0]
        ).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("indices".to_string(), Value::Tensor(indices_data));
        inputs.insert("emb_weight".to_string(), Value::Tensor(weight_data));

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
                        crate::asg::NodeType::Input { name: n } |
                        crate::asg::NodeType::Parameter { name: n } if n == &name
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
            // Output shape: [2, 3, 4]
            assert_eq!(arr.shape(), &[2, 3, 4]);
        } else {
            panic!("Expected tensor output");
        }
    }
}
