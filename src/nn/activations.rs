//! Модуль, содержащий слои-активации, такие как ReLU,
//! реализованные для графовой архитектуры.

use crate::nn::module::Module;
use crate::tensor::Tensor;

// --- Слой ReLU ---

/// Слой активации ReLU (Rectified Linear Unit) в графовой парадигме.
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
    /// Прямой проход добавляет в граф узел `ReLU`.
    fn forward(&self, inputs: &Tensor) -> Tensor {
        inputs.relu()
    }

    /// ReLU не имеет обучаемых параметров, поэтому возвращаем пустой вектор.
    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}