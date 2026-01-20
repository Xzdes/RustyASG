// --- Файл: src/nn/pooling.rs ---

//! Модуль, реализующий слои пулинга (pooling) для CNN.

use crate::nn::module::Module;
use crate::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Max Pooling 2D слой.
///
/// Применяет max pooling к входному тензору формы [N, C, H, W].
/// Выбирает максимальное значение из каждого окна.
///
/// # Пример
///
/// ```rust,ignore
/// let pool = MaxPool2d::new((2, 2), (2, 2));
/// let output = pool.forward(&input); // Уменьшает H и W в 2 раза
/// ```
pub struct MaxPool2d {
    /// Размер окна (kH, kW).
    pub kernel_size: (usize, usize),
    /// Шаг (stride_h, stride_w).
    pub stride: (usize, usize),
}

impl MaxPool2d {
    /// Создает MaxPool2d слой.
    ///
    /// # Аргументы
    ///
    /// * `kernel_size` - Размер окна пулинга
    /// * `stride` - Шаг пулинга (обычно равен kernel_size)
    pub fn new(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self { kernel_size, stride }
    }

    /// Создает MaxPool2d с одинаковым kernel_size и stride.
    pub fn square(size: usize) -> Self {
        Self {
            kernel_size: (size, size),
            stride: (size, size),
        }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        inputs.max_pool2d(self.kernel_size, self.stride)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![] // Pooling слои не имеют обучаемых параметров
    }
}

/// Average Pooling 2D слой.
///
/// Применяет average pooling к входному тензору формы [N, C, H, W].
/// Вычисляет среднее значение в каждом окне.
pub struct AvgPool2d {
    /// Размер окна (kH, kW).
    pub kernel_size: (usize, usize),
    /// Шаг (stride_h, stride_w).
    pub stride: (usize, usize),
    /// Паддинг (pad_h, pad_w).
    pub padding: (usize, usize),
}

impl AvgPool2d {
    /// Создает AvgPool2d слой.
    pub fn new(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride,
            padding: (0, 0),
        }
    }

    /// Создает AvgPool2d с одинаковым kernel_size и stride.
    pub fn square(size: usize) -> Self {
        Self {
            kernel_size: (size, size),
            stride: (size, size),
            padding: (0, 0),
        }
    }

    /// Устанавливает паддинг.
    pub fn with_padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }
}

impl Module for AvgPool2d {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        inputs.avg_pool2d(self.kernel_size, self.stride, self.padding)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Adaptive Average Pooling 2D.
///
/// Автоматически вычисляет параметры пулинга для достижения
/// заданного выходного размера. Полезен для создания фиксированного
/// размера выхода независимо от размера входа.
///
/// # Пример
///
/// ```rust,ignore
/// // Всегда выдает выход размера [N, C, 1, 1] независимо от размера входа
/// let gap = AdaptiveAvgPool2d::new((1, 1));
/// let output = gap.forward(&input); // Global Average Pooling
/// ```
pub struct AdaptiveAvgPool2d {
    /// Целевой выходной размер (H_out, W_out).
    pub output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    /// Создает AdaptiveAvgPool2d слой.
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }

    /// Создает Global Average Pooling (GAP) - выходной размер (1, 1).
    pub fn global() -> Self {
        Self { output_size: (1, 1) }
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        inputs.adaptive_avg_pool2d(self.output_size)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/// Global Average Pooling - сокращение AdaptiveAvgPool2d с output_size (1, 1).
pub type GlobalAvgPool2d = AdaptiveAvgPool2d;

impl GlobalAvgPool2d {
    /// Создает Global Average Pooling слой.
    pub fn new_global() -> Self {
        AdaptiveAvgPool2d::global()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_pool2d() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let input = Tensor::new_input(&context, "input");
        let pool = MaxPool2d::new((2, 2), (2, 2));

        let output = pool.forward(&input);
        assert!(pool.parameters().is_empty());
    }

    #[test]
    fn test_max_pool2d_square() {
        let pool = MaxPool2d::square(2);
        assert_eq!(pool.kernel_size, (2, 2));
        assert_eq!(pool.stride, (2, 2));
    }

    #[test]
    fn test_avg_pool2d() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let input = Tensor::new_input(&context, "input");
        let pool = AvgPool2d::new((2, 2), (2, 2)).with_padding((1, 1));

        let output = pool.forward(&input);
        assert_eq!(pool.padding, (1, 1));
    }

    #[test]
    fn test_adaptive_avg_pool2d() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let input = Tensor::new_input(&context, "input");
        let pool = AdaptiveAvgPool2d::new((7, 7));

        let output = pool.forward(&input);
        assert_eq!(pool.output_size, (7, 7));
    }

    #[test]
    fn test_global_avg_pool() {
        let gap = AdaptiveAvgPool2d::global();
        assert_eq!(gap.output_size, (1, 1));
    }
}
