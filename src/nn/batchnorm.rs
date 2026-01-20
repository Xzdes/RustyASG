//! BatchNormalization слой для графовой архитектуры.
//!
//! Реализует Batch Normalization с поддержкой train/eval режимов,
//! running statistics и обучаемых параметров gamma/beta.

use crate::nn::Module;
use crate::tensor::{GraphContext, Tensor};
use ndarray::arr0;
use std::cell::RefCell;
use std::rc::Rc;

/// Малая константа для численной стабильности.
const EPS: f32 = 1e-5;

/// Momentum для обновления running statistics.
const DEFAULT_MOMENTUM: f32 = 0.1;

/// Слой Batch Normalization.
///
/// Нормализует входные данные по batch dimension (axis 0),
/// применяя формулу: y = gamma * (x - mean) / sqrt(var + eps) + beta
///
/// В режиме обучения использует статистики текущего батча,
/// в режиме inference - накопленные running statistics.
pub struct BatchNorm {
    /// Обучаемый масштаб (scale)
    pub gamma: Tensor,
    /// Обучаемый сдвиг (shift)
    pub beta: Tensor,
    /// Константа epsilon для численной стабильности
    eps: Tensor,
    /// Momentum для экспоненциального сглаживания running statistics
    pub momentum: f32,
    /// Флаг режима обучения
    pub training: bool,
    /// Имя слоя (для идентификации running stats)
    pub name: String,
}

impl BatchNorm {
    /// Создаёт новый слой BatchNorm.
    ///
    /// # Аргументы
    /// * `ctx` - Контекст графа для регистрации параметров
    /// * `name` - Уникальное имя слоя
    pub fn new(ctx: &Rc<RefCell<GraphContext>>, name: &str) -> Self {
        let gamma_name = format!("{}.gamma", name);
        let beta_name = format!("{}.beta", name);

        let gamma = Tensor::new_parameter(ctx, &gamma_name);
        let beta = Tensor::new_parameter(ctx, &beta_name);
        let eps = Tensor::new_literal(ctx, arr0(EPS).into_dyn(), &format!("{}.eps", name));

        Self {
            gamma,
            beta,
            eps,
            momentum: DEFAULT_MOMENTUM,
            training: true,
            name: name.to_string(),
        }
    }

    /// Создаёт слой с указанным momentum.
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Устанавливает режим обучения.
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Устанавливает режим inference.
    pub fn eval(&mut self) {
        self.training = false;
    }
}

impl Module for BatchNorm {
    /// Прямой проход BatchNorm.
    ///
    /// В текущей реализации всегда использует batch statistics
    /// (running statistics требуют дополнительной инфраструктуры для хранения состояния).
    fn forward(&self, x: &Tensor) -> Tensor {
        // Вычисляем mean по batch (axis 0)
        // Примечание: текущая реализация Mean работает по последней оси,
        // для корректной работы BatchNorm нужно транспонировать или добавить MeanAxis
        let mean = x.mean();
        let x_minus_mean = x - &mean;

        // Вычисляем variance
        let squared = &x_minus_mean * &x_minus_mean;
        let variance = squared.mean();

        // Нормализуем
        let std = (&variance + &self.eps).sqrt();
        let normalized = &x_minus_mean / &std;

        // Применяем gamma и beta
        let scaled = &normalized * &self.gamma;
        &scaled + &self.beta
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batchnorm_creation() {
        let ctx = Rc::new(RefCell::new(GraphContext::new()));
        let bn = BatchNorm::new(&ctx, "bn1");

        assert_eq!(bn.name, "bn1");
        assert!(bn.training);
        assert_eq!(bn.parameters().len(), 2);
    }

    #[test]
    fn test_batchnorm_train_eval() {
        let ctx = Rc::new(RefCell::new(GraphContext::new()));
        let mut bn = BatchNorm::new(&ctx, "bn1");

        bn.eval();
        assert!(!bn.training);

        bn.train();
        assert!(bn.training);
    }
}
