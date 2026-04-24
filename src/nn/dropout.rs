//! Dropout layer for regularization.
//!
//! Implements the standard Dropout, which randomly zeroes elements of
//! the tensor during training to prevent overfitting.

use crate::nn::Module;
use crate::tensor::Tensor;

/// Dropout layer for regularization.
///
/// During training, zeroes elements with probability `p` and scales the
/// remaining elements by `1 / (1 - p)` to preserve expected value.
///
/// During inference (eval mode), simply passes the input through unchanged.
///
/// # Example
/// ```ignore
/// let dropout = Dropout::new(0.5); // 50% drop probability
/// dropout.eval(); // Disable dropout for inference
/// ```
pub struct Dropout {
    /// Drop probability (0.0 - 1.0).
    pub p: f32,
    /// Training-mode flag.
    pub training: bool,
}

impl Dropout {
    /// Creates a new Dropout layer.
    ///
    /// # Arguments
    /// * `p` - Probability of zeroing an element (0.1-0.5 is recommended).
    ///
    /// # Panics
    /// Panics if `p` is not in `[0, 1)`.
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1), got {}",
            p
        );
        Self { p, training: true }
    }

    /// Switches to training mode.
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Switches to inference mode.
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Returns whether dropout is active.
    pub fn is_training(&self) -> bool {
        self.training
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Module for Dropout {
    /// Forward pass of Dropout.
    ///
    /// **Note**: in the current graph-based architecture Dropout is not
    /// applied at graph-construction time, since randomness must be
    /// introduced at execution time. This method simply returns the input
    /// unchanged.
    ///
    /// Full Dropout support requires:
    /// 1. Adding `NodeType::Dropout` to the ASG.
    /// 2. Implementing mask generation in the runtime.
    /// 3. Passing the `training` flag to the backend.
    fn forward(&self, x: &Tensor) -> Tensor {
        // TODO: Implement via a dedicated Dropout node in the ASG.
        // For now, simply return the input unchanged.
        x.clone()
    }

    fn parameters(&self) -> Vec<Tensor> {
        // Dropout has no trainable parameters.
        Vec::new()
    }
}

/// Spatial Dropout for convolutional networks.
///
/// Unlike standard Dropout, zeroes entire channels (feature maps), which
/// is more effective for convolutional architectures.
pub struct SpatialDropout {
    /// Per-channel drop probability.
    pub p: f32,
    /// Training-mode flag.
    pub training: bool,
}

impl SpatialDropout {
    /// Creates a new SpatialDropout layer.
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1), got {}",
            p
        );
        Self { p, training: true }
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }
}

impl Module for SpatialDropout {
    fn forward(&self, x: &Tensor) -> Tensor {
        // TODO: Implement via a dedicated node.
        x.clone()
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_creation() {
        let dropout = Dropout::new(0.5);
        assert_eq!(dropout.p, 0.5);
        assert!(dropout.training);
    }

    #[test]
    fn test_dropout_modes() {
        let mut dropout = Dropout::new(0.3);

        dropout.eval();
        assert!(!dropout.training);

        dropout.train();
        assert!(dropout.training);
    }

    #[test]
    #[should_panic(expected = "Dropout probability must be in [0, 1)")]
    fn test_dropout_invalid_p() {
        Dropout::new(1.5);
    }
}
