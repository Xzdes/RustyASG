// --- Файл: src/nn/norm.rs (финальная рабочая версия для замены) ---

//! Модуль, реализующий слой нормализации (Layer Normalization) для графовой архитектуры.

use crate::nn::module::Module;
use crate::tensor::{GraphContext, Tensor};
use ndarray::arr0;
use std::cell::RefCell;
use std::rc::Rc;

/// Малая константа для численной стабильности при делении на стандартное отклонение.
const EPSILON: f32 = 1e-5;

/// Слой нормализации (Layer Normalization).
///
/// Нормализует активации по оси признаков (последней оси) для каждого элемента в батче.
/// Имеет два обучаемых параметра: `gamma` (масштаб) и `beta` (сдвиг).
pub struct LayerNorm {
    /// Обучаемый параметр масштабирования (gain). Инициализируется единицами.
    pub gamma: Tensor,
    /// Обучаемый параметр сдвига (bias). Инициализируется нулями.
    pub beta: Tensor,
    /// Малая константа для избежания деления на ноль, представленная как узел в графе.
    pub epsilon: Tensor,
}

impl LayerNorm {
    /// Создает новый слой LayerNorm.
    pub fn new(context: &Rc<RefCell<GraphContext>>, name: &str) -> Self {
        let gamma_name = format!("{}.gamma", name);
        let beta_name = format!("{}.beta", name);

        let gamma = Tensor::new_parameter(context, &gamma_name);
        let beta = Tensor::new_parameter(context, &beta_name);
        
        let epsilon = Tensor::new_literal(
            context,
            arr0(EPSILON).into_dyn(),
            &format!("{}.epsilon", name),
        );

        Self {
            gamma,
            beta,
            epsilon,
        }
    }
}

impl Module for LayerNorm {
    /// Выполняет прямой проход LayerNorm, строя канонический, численно стабильный подграф.
    fn forward(&self, inputs: &Tensor) -> Tensor {
        let mean = inputs.mean();
        let variance = inputs.variance();

        // Стандартная реализация: y = (x - mean) / sqrt(variance + epsilon)
        let denominator = (&variance + &self.epsilon).sqrt();
        let normalized = &(inputs - &mean) / &denominator;

        // Применяем обучаемые параметры
        let scaled = &normalized * &self.gamma;
        let shifted = &scaled + &self.beta;

        shifted
    }

    /// Возвращает `gamma` и `beta` как обучаемые параметры.
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}