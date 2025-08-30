//! Модуль, реализующий слой FeedForward, адаптированный для графовой архитектуры.

use crate::nn::{Linear, Module, ReLU};
use crate::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Слой FeedForward, стандартный компонент блока Трансформера.
///
/// Состоит из двух линейных слоев с активацией ReLU между ними.
/// Этот слой применяется к каждой позиции в последовательности независимо.
///
/// В графовой архитектуре он конструирует соответствующий подграф операций.
/// Формула: `FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂`
pub struct FeedForward {
    /// Первый линейный слой, расширяющий размерность.
    linear1: Linear,
    /// Слой активации ReLU.
    relu: ReLU,
    /// Второй линейный слой, сжимающий размерность обратно к исходной.
    linear2: Linear,
}

impl FeedForward {
    /// Создает новый слой FeedForward, регистрируя его параметры в графе.
    ///
    /// # Аргументы
    ///
    /// * `context` - Ссылка на `GraphContext`, в котором будет строиться граф.
    /// * `embed_dim` - Размерность входных и выходных векторов (например, 512).
    /// * `hidden_dim` - Размерность скрытого слоя (обычно в 4 раза больше `embed_dim`, например, 2048).
    /// * `name` - Базовое имя для этого слоя, чтобы параметры имели уникальные имена.
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        embed_dim: usize,
        hidden_dim: usize,
        name: &str,
    ) -> Self {
        let linear1_name = format!("{}.linear1", name);
        let linear2_name = format!("{}.linear2", name);

        Self {
            linear1: Linear::new(context, embed_dim, hidden_dim, &linear1_name),
            relu: ReLU::new(),
            linear2: Linear::new(context, hidden_dim, embed_dim, &linear2_name),
        }
    }
}

impl Module for FeedForward {
    /// Прямой проход: `Linear -> ReLU -> Linear`.
    /// Добавляет в граф соответствующие узлы операций.
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