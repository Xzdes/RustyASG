//! Модуль, содержащий слои-активации, такие как ReLU и Sigmoid.

use crate::nn::module::Module;
use crate::tensor::Tensor;
// --- ИЗМЕНЕНИЕ: Импортируем наш Result ---
use crate::error::Result;

/// Слой активации ReLU (Rectified Linear Unit).
///
/// Применяет поэлементную функцию `max(0, x)`.
/// Этот слой не имеет обучаемых параметров.
pub struct ReLU;

impl ReLU {
    /// Создает новый экземпляр слоя ReLU.
    pub fn new() -> Self {
        ReLU {}
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ReLU {
    /// Прямой проход просто вызывает математическую операцию ReLU.
    // --- ИЗМЕНЕНИЕ: Сигнатура функции обновлена ---
    fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        // Мы уже добавили метод .relu() в Tensor.
        // Поскольку `relu_op` пока не возвращает Result, мы оборачиваем
        // результат в `Ok()`, чтобы соответствовать новому трейту.
        // --- ИЗМЕНЕНИЕ: Результат обернут в Ok() ---
        Ok(inputs.relu())
    }

    /// ReLU не имеет обучаемых параметров, поэтому возвращаем пустой вектор.
    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

// --- НОВЫЙ СЛОЙ: Sigmoid ---

/// Слой активации Sigmoid.
///
/// "Сжимает" входные значения в диапазон (0, 1), что полезно для
/// интерпретации выходов как вероятностей.
/// Применяет поэлементную функцию `1 / (1 + e^(-x))`.
pub struct Sigmoid;

impl Sigmoid {
    /// Создает новый экземпляр слоя Sigmoid.
    pub fn new() -> Self {
        Sigmoid {}
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sigmoid {
    /// Прямой проход вызывает математическую операцию Sigmoid.
    // --- ИЗМЕНЕНИЕ: Сигнатура функции обновлена ---
    fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        // --- ИЗМЕНЕНИЕ: Результат обернут в Ok() ---
        Ok(crate::ops::elementwise::sigmoid_op(inputs))
    }

    /// Sigmoid не имеет обучаемых параметров.
    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}