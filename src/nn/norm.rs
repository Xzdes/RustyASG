//! Модуль, реализующий слой нормализации (Layer Normalization).

use crate::nn::module::Module;
use crate::tensor::Tensor;
use crate::error::Result;

/// Малая константа для численной стабильности при делении на стандартное отклонение.
const EPSILON: f32 = 1e-5;

/// Слой нормализации (Layer Normalization).
///
/// Нормализует активации по оси признаков (последней оси) для каждого элемента в батче.
/// В отличие от BatchNorm, его вычисления полностью независимы для каждого элемента
/// в батче, что делает его популярным в моделях обработки последовательностей,
/// таких как Трансформеры.
///
/// Имеет два обучаемых параметра: `gamma` (масштаб) и `beta` (сдвиг), которые позволяют
/// сети восстановить исходное распределение, если это необходимо.
pub struct LayerNorm {
    /// Обучаемый параметр масштабирования (gain). Инициализируется единицами.
    gamma: Tensor,
    /// Обучаемый параметр сдвига (bias). Инициализируется нулями.
    beta: Tensor,
    /// Малая константа для избежания деления на ноль.
    epsilon: f32,
}

impl LayerNorm {
    /// Создает новый слой LayerNorm.
    ///
    /// # Аргументы
    ///
    /// * `normalized_shape` - Размерность признаков, по которой происходит нормализация.
    ///   Например, для Трансформера это будет `embedding_dim`.
    pub fn new(normalized_shape: usize) -> Self {
        // gamma инициализируется единицами. Форма [1, normalized_shape] для broadcasting'а.
        let gamma = Tensor::ones(&[1, normalized_shape], true);
        // beta инициализируется нулями.
        let beta = Tensor::zeros(&[1, normalized_shape], true);

        Self {
            gamma,
            beta,
            epsilon: EPSILON,
        }
    }
}

impl Module for LayerNorm {
    /// Выполняет прямой проход LayerNorm.
    ///
    /// Формула: `y = (x - mean(x)) / sqrt(var(x) + epsilon) * gamma + beta`
    fn forward(&self, inputs: &Tensor) -> Result<Tensor> {
        // --- ИЗМЕНЕНИЕ: Убираем временный `Ok()` ---
        // Теперь `layernorm_op` сама возвращает `Result`.
        inputs.layer_norm(&self.gamma, &self.beta, self.epsilon)
    }

    /// Возвращает `gamma` и `beta` как обучаемые параметры.
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}