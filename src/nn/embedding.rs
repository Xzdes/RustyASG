//! Модуль, реализующий слой встраивания (Embedding).

use crate::nn::module::Module;
use crate::tensor::Tensor;
use crate::error::Result;
use ndarray::IxDyn;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

/// Слой встраивания (Embedding Layer).
///
/// Хранит таблицу векторов (встраиваний) и по заданным целочисленным индексам
/// (например, ID токенов из словаря) возвращает соответствующие векторы.
/// Это основной способ представления дискретных переменных (как слова) в виде
/// непрерывных векторов для нейронной сети.
pub struct Embedding {
    /// Матрица весов, где каждая строка - это вектор-встраивание.
    /// Форма: `[vocab_size, embedding_dim]`.
    pub weights: Tensor,
    /// Размер словаря (количество уникальных токенов).
    pub vocab_size: usize,
    /// Размерность вектора-встраивания для каждого токена.
    pub embedding_dim: usize,
}

impl Embedding {
    /// Создает новый слой Embedding.
    ///
    /// # Аргументы
    ///
    /// * `vocab_size` - Размер словаря (количество уникальных токенов/элементов).
    /// * `embedding_dim` - Размерность вектора-встраивания для каждого токена.
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        // Инициализируем веса небольшими случайными значениями из равномерного распределения.
        let weights_data = ndarray::ArrayD::random(
            IxDyn(&[vocab_size, embedding_dim]),
            Uniform::new(-1.0, 1.0),
        );
        let weights = Tensor::new(weights_data, true);

        Self {
            weights,
            vocab_size,
            embedding_dim,
        }
    }
}

impl Module for Embedding {
    /// Выполняет прямой проход, выбирая векторы по индексам.
    ///
    /// # Аргументы
    ///
    /// * `inputs` - Тензор с целочисленными индексами (ID токенов).
    ///   Ожидаемая форма: `[batch_size, seq_len]`.
    ///
    /// # Возвращает
    ///
    /// Тензор с соответствующими встраиваниями.
    /// Выходная форма: `[batch_size, seq_len, embedding_dim]`.
    fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        inputs.embedding(&self.weights)
    }

    /// Возвращает матрицу весов как единственный обучаемый параметр.
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone()]
    }
}