//! Activation layers for the graph-based architecture.
//!
//! Includes implementations of popular activation functions:
//! - ReLU (Rectified Linear Unit)
//! - LeakyReLU
//! - GELU (Gaussian Error Linear Unit)
//! - SiLU/Swish
//! - Tanh
//! - Sigmoid
//! - ELU (Exponential Linear Unit)
//! - Softplus

use crate::nn::module::Module;
use crate::tensor::Tensor;

// ============================================================================
// ReLU - Rectified Linear Unit
// ============================================================================

/// ReLU activation layer: `f(x) = max(0, x)`.
///
/// The most popular activation function for hidden layers. It is
/// computationally cheap and works well in practice.
pub struct ReLU;

impl ReLU {
    pub fn new() -> Self {
        ReLU {}
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for ReLU {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        inputs.relu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

// ============================================================================
// LeakyReLU
// ============================================================================

/// LeakyReLU activation layer: `f(x) = x if x > 0 else alpha * x`.
///
/// Addresses the "dying ReLU" problem by allowing a small gradient for
/// negative values.
///
/// Common in GAN architectures.
pub struct LeakyReLU {
    /// Slope for negative values (usually 0.01 or 0.2).
    pub negative_slope: f32,
}

impl LeakyReLU {
    /// Creates a LeakyReLU with the given negative slope.
    pub fn new(negative_slope: f32) -> Self {
        Self { negative_slope }
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl Module for LeakyReLU {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        // LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)
        // = relu(x) - negative_slope * relu(-x)
        // Equivalently: x * (x > 0) + negative_slope * x * (x <= 0)
        inputs.leaky_relu(self.negative_slope)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

// ============================================================================
// GELU - Gaussian Error Linear Unit
// ============================================================================

/// GELU activation layer: `f(x) = x * Φ(x)`.
///
/// where `Φ(x)` is the CDF of the standard normal distribution.
///
/// Used in BERT, GPT and other transformer architectures. Provides a
/// smooth nonlinearity.
pub struct GELU;

impl GELU {
    pub fn new() -> Self {
        GELU {}
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for GELU {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        inputs.gelu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

// ============================================================================
// SiLU / Swish
// ============================================================================

/// SiLU activation layer (Sigmoid Linear Unit), also known as Swish.
///
/// `f(x) = x * sigmoid(x)`
///
/// Used in EfficientNet, Mish and other modern architectures. A smooth
/// non-monotonic function with good optimization properties.
pub struct SiLU;

impl SiLU {
    pub fn new() -> Self {
        SiLU {}
    }
}

impl Default for SiLU {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for SiLU {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        inputs.silu()
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

/// Alias for SiLU - historically known as Swish.
pub type Swish = SiLU;

// ============================================================================
// Tanh
// ============================================================================

/// Tanh activation layer: `f(x) = tanh(x) = (e^x - e^-x) / (e^x + e^-x)`.
///
/// Classic S-shaped function with output in `(-1, 1)`. Commonly used in
/// RNN/LSTM networks.
pub struct Tanh;

impl Tanh {
    pub fn new() -> Self {
        Tanh {}
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Tanh {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        inputs.tanh()
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

// ============================================================================
// Sigmoid
// ============================================================================

/// Sigmoid activation layer: `f(x) = 1 / (1 + e^-x)`.
///
/// Classic S-shaped function with output in `(0, 1)`. Used for binary
/// classification and gating mechanisms.
pub struct Sigmoid;

impl Sigmoid {
    pub fn new() -> Self {
        Sigmoid {}
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sigmoid {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        inputs.sigmoid()
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

// ============================================================================
// ELU - Exponential Linear Unit
// ============================================================================

/// ELU activation layer: `f(x) = x if x > 0 else alpha * (e^x - 1)`.
///
/// A smooth alternative to ReLU that allows negative values. Helps
/// speed up training and improve generalization.
pub struct ELU {
    /// Alpha parameter for negative values (usually 1.0).
    pub alpha: f32,
}

impl ELU {
    pub fn new(alpha: f32) -> Self {
        Self { alpha }
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Module for ELU {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        inputs.elu(self.alpha)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

// ============================================================================
// Softplus
// ============================================================================

/// Softplus activation layer: `f(x) = log(1 + e^x)`.
///
/// A smooth approximation of ReLU. Always positive. Its derivative is
/// `sigmoid(x)`.
pub struct Softplus {
    /// Beta parameter (default 1.0).
    pub beta: f32,
    /// Threshold for numerical stability.
    pub threshold: f32,
}

impl Softplus {
    pub fn new(beta: f32, threshold: f32) -> Self {
        Self { beta, threshold }
    }
}

impl Default for Softplus {
    fn default() -> Self {
        Self::new(1.0, 20.0)
    }
}

impl Module for Softplus {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        inputs.softplus(self.beta)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

// ============================================================================
// Softmax
// ============================================================================

/// Softmax layer: `f(x)_i = e^x_i / sum(e^x_j)`.
///
/// Converts a vector into a probability distribution. Used for
/// multi-class classification.
pub struct Softmax;

impl Softmax {
    pub fn new() -> Self {
        Softmax {}
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Softmax {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        inputs.softmax()
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}
