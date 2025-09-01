//! Модуль, реализующий слой FeedForward, адаптированный для графовой архитектуры.

use crate::nn::{Linear, Module, ReLU};
use crate::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Слой FeedForward, стандартный компонент блока Трансформера.
///
/// Состоит из двух линейных слоев с активацией ReLU между ними.
pub struct FeedForward {
    /// Первый линейный слой.
    linear1: Linear,
    /// Слой активации ReLU.
    relu: ReLU,
    /// Второй линейный слой.
    linear2: Linear,
}

impl FeedForward {
    /// Создает новый слой FeedForward, регистрируя его параметры в графе.
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        name: &str,
    ) -> Self {
        let linear1_name = format!("{}.linear1", name);
        let linear2_name = format!("{}.linear2", name);

        Self {
            linear1: Linear::new(context, &linear1_name),
            relu: ReLU::new(),
            linear2: Linear::new(context, &linear2_name),
        }
    }
}

impl Module for FeedForward {
    /// Прямой проход: `Linear -> ReLU -> Linear`.
    fn forward(&self, inputs: &Tensor) -> Tensor {
        let x = self.linear1.forward(inputs);
        let x = self.relu.forward(&x);
        self.linear2.forward(&x)
    }

    /// Собирает параметры из обоих вложенных линейных слоев.
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params
    }
}