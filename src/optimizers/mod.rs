//! Модуль, содержащий реализации оптимизаторов для обновления весов модели.
//!
//! Оптимизаторы работают с реальными числовыми данными (`Value`) на CPU.
//! Они получают текущие веса и вычисленные градиенты и применяют
//! алгоритм обновления.

use crate::asg::Value;
use std::collections::HashMap;

/// Трейт, определяющий общий интерфейс для всех оптимизаторов.
pub trait Optimizer {
    /// Выполняет один шаг оптимизации, обновляя веса.
    ///
    /// # Аргументы
    ///
    /// * `parameters` - Изменяемый `HashMap`, содержащий текущие значения весов модели.
    ///   Ключ - это имя параметра (например, "layer1.weights").
    /// * `gradients` - `HashMap`, содержащий вычисленные значения градиентов для этих весов.
    fn step(&self, parameters: &mut HashMap<String, Value>, gradients: &HashMap<String, Value>);
}

/// Реализация оптимизатора Stochastic Gradient Descent (SGD).
pub struct Sgd {
    /// Скорость обучения (learning rate).
    lr: f32,
}

impl Sgd {
    /// Создает новый экземпляр оптимизатора SGD.
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl Optimizer for Sgd {
    fn step(&self, parameters: &mut HashMap<String, Value>, gradients: &HashMap<String, Value>) {
        // Проходим по всем градиентам, которые были вычислены.
        for (param_name, grad_value) in gradients {
            // Получаем текущее значение параметра и его градиент.
            if let (Some(Value::Tensor(param_value)), Value::Tensor(grad_tensor)) =
                (parameters.get_mut(param_name), grad_value)
            {
                // Выполняем обновление весов: param = param - lr * grad
                ndarray::azip!((p in param_value, &g in grad_tensor) *p = *p - self.lr * g);
            }
        }
    }
}