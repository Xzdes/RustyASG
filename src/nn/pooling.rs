// --- File: src/nn/pooling.rs ---

//! Module implementing pooling layers for CNNs.

use crate::nn::module::Module;
use crate::tensor::Tensor;

/// Max Pooling 2D layer.
///
/// Applies max pooling to input tensor of shape [N, C, H, W].
/// Selects maximum value from each window.
///
/// # Example
///
/// ```rust,ignore
/// let pool = MaxPool2d::new((2, 2), (2, 2));
/// let output = pool.forward(&input); // Reduces H and W by half
/// ```
pub struct MaxPool2d {
    /// Window size (kH, kW).
    pub kernel_size: (usize, usize),
    /// Stride (stride_h, stride_w).
    pub stride: (usize, usize),
}

impl MaxPool2d {
    /// Creates MaxPool2d layer.
    ///
    /// # Arguments
    ///
    /// * `kernel_size` - Pooling window size
    /// * `stride` - Pooling stride (usually equals kernel_size)
    pub fn new(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self { kernel_size, stride }
    }

    /// Creates MaxPool2d with equal kernel_size and stride.
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
        vec![] // Pooling layers have no trainable parameters
    }
}

/// Average Pooling 2D layer.
///
/// Applies average pooling to input tensor of shape [N, C, H, W].
/// Computes average value in each window.
pub struct AvgPool2d {
    /// Window size (kH, kW).
    pub kernel_size: (usize, usize),
    /// Stride (stride_h, stride_w).
    pub stride: (usize, usize),
    /// Padding (pad_h, pad_w).
    pub padding: (usize, usize),
}

impl AvgPool2d {
    /// Creates AvgPool2d layer.
    pub fn new(kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        Self {
            kernel_size,
            stride,
            padding: (0, 0),
        }
    }

    /// Creates AvgPool2d with equal kernel_size and stride.
    pub fn square(size: usize) -> Self {
        Self {
            kernel_size: (size, size),
            stride: (size, size),
            padding: (0, 0),
        }
    }

    /// Sets padding.
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
/// Automatically computes pooling parameters to achieve
/// target output size. Useful for creating fixed output
/// size regardless of input size.
///
/// # Example
///
/// ```rust,ignore
/// // Always outputs size [N, C, 1, 1] regardless of input size
/// let gap = AdaptiveAvgPool2d::new((1, 1));
/// let output = gap.forward(&input); // Global Average Pooling
/// ```
pub struct AdaptiveAvgPool2d {
    /// Target output size (H_out, W_out).
    pub output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    /// Creates AdaptiveAvgPool2d layer.
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }

    /// Creates Global Average Pooling (GAP) - output size (1, 1).
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

/// Global Average Pooling - shorthand for AdaptiveAvgPool2d with output_size (1, 1).
pub type GlobalAvgPool2d = AdaptiveAvgPool2d;

impl GlobalAvgPool2d {
    /// Creates Global Average Pooling layer.
    pub fn new_global() -> Self {
        AdaptiveAvgPool2d::global()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::GraphContext;
    use std::cell::RefCell;
    use std::rc::Rc;

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
