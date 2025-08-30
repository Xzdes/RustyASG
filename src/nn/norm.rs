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
    gamma: Tensor,
    /// Обучаемый параметр сдвига (bias). Инициализируется нулями.
    beta: Tensor,
    /// Малая константа для избежания деления на ноль.
    epsilon: Tensor,
}

impl LayerNorm {
    /// Создает новый слой LayerNorm.
    ///
    /// # Аргументы
    ///
    /// * `context` - Контекст графа, в котором будут созданы параметры.
    /// * `name` - Уникальное имя для параметров слоя.
    pub fn new(context: &Rc<RefCell<GraphContext>>, name: &str) -> Self {
        let gamma_name = format!("{}.gamma", name);
        let beta_name = format!("{}.beta", name);

        // gamma и beta - это обучаемые параметры.
        let gamma = Tensor::new_parameter(context, &gamma_name);
        let beta = Tensor::new_parameter(context, &beta_name);
        
        // epsilon - это константа, встраиваемая в граф.
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
    /// Выполняет прямой проход LayerNorm, строя соответствующий подграф.
    ///
    /// Формула: `y = (x - mean(x)) / sqrt(var(x) + epsilon) * gamma + beta`
    fn forward(&self, inputs: &Tensor) -> Tensor {
        let mean = inputs.mean();
        let variance = inputs.variance();

        let x_minus_mean = inputs - &mean;
        let var_plus_eps = &variance + &self.epsilon;
        let std_dev = var_plus_eps.sqrt();
        
        let normalized = &x_minus_mean / &std_dev;

        let scaled = &normalized * &self.gamma;
        let shifted = &scaled + &self.beta;

        shifted
    }

    /// Возвращает `gamma` и `beta` как обучаемые параметры.
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}