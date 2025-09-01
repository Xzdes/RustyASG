//! Модуль, реализующий полносвязный (линейный) слой в графовой парадигме.

use crate::nn::module::Module;
use crate::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Полносвязный (линейный) слой.
///
/// В графовой архитектуре этот слой не хранит реальных данных. Вместо этого
/// он владеет символьными `Tensor`-дескрипторами, которые представляют его
/// веса (`weights`) и смещения (`bias`) как узлы `Parameter` в ASG.
///
/// Метод `forward` добавляет в граф операции, соответствующие формуле `y = xW + b`.
pub struct Linear {
    /// Символьный дескриптор для тензора весов.
    pub weights: Tensor,
    /// Символьный дескриптор для тензора смещений.
    pub bias: Tensor,
}

impl Linear {
    /// Создает новый полносвязный слой, регистрируя его параметры в графе.
    ///
    /// # Аргументы
    ///
    /// * `context` - Ссылка на `GraphContext`, в котором будет строиться граф.
    /// * `name` - Базовое имя для этого слоя, чтобы параметры имели уникальные
    ///   имена в графе (например, "layer1.weights", "layer1.bias").
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        name: &str,
    ) -> Self {
        // Создаем символьные узлы-параметры в графе.
        // Реальные значения и формы для этих параметров будут предоставлены
        // позже, перед запуском выполнения и shape inference.
        let weights_name = format!("{}.weights", name);
        let bias_name = format!("{}.bias", name);

        let weights = Tensor::new_parameter(context, &weights_name);
        let bias = Tensor::new_parameter(context, &bias_name);

        Self { weights, bias }
    }
}

impl Module for Linear {
    /// Добавляет в граф операции для прямого прохода через линейный слой.
    ///
    /// Конструирует подграф, соответствующий `inputs.dot(weights) + bias`.
    fn forward(&self, inputs: &Tensor) -> Tensor {
        let dot_product = inputs.dot(&self.weights);
        let final_output = &dot_product + &self.bias;
        final_output
    }

    /// Возвращает список символьных дескрипторов для обучаемых параметров слоя.
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}