//! Модуль, содержащий слои-активации для графовой архитектуры.
//!
//! Включает реализации популярных функций активации:
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

/// Слой активации ReLU: `f(x) = max(0, x)`
///
/// Самая популярная функция активации для скрытых слоев.
/// Вычислительно эффективна и хорошо работает на практике.
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

/// Слой активации LeakyReLU: `f(x) = x if x > 0 else alpha * x`
///
/// Решает проблему "умирающих нейронов" ReLU, позволяя небольшой
/// градиент для отрицательных значений.
///
/// Популярен в GAN архитектурах.
pub struct LeakyReLU {
    /// Наклон для отрицательных значений (обычно 0.01 или 0.2)
    pub negative_slope: f32,
}

impl LeakyReLU {
    /// Создаёт LeakyReLU с указанным наклоном.
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
        // Для упрощения используем: x * (x > 0) + negative_slope * x * (x <= 0)
        inputs.leaky_relu(self.negative_slope)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

// ============================================================================
// GELU - Gaussian Error Linear Unit
// ============================================================================

/// Слой активации GELU: `f(x) = x * Φ(x)`
///
/// где Φ(x) - CDF стандартного нормального распределения.
///
/// Используется в BERT, GPT и других трансформерных архитектурах.
/// Обеспечивает гладкую нелинейность.
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

/// Слой активации SiLU (Sigmoid Linear Unit), также известный как Swish.
///
/// `f(x) = x * sigmoid(x)`
///
/// Используется в EfficientNet, Mish и других современных архитектурах.
/// Гладкая, не-монотонная функция с хорошими свойствами оптимизации.
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

/// Алиас для SiLU - исторически известен как Swish
pub type Swish = SiLU;

// ============================================================================
// Tanh
// ============================================================================

/// Слой активации Tanh: `f(x) = tanh(x) = (e^x - e^-x) / (e^x + e^-x)`
///
/// Классическая S-образная функция с выходом в диапазоне (-1, 1).
/// Часто используется в RNN/LSTM.
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

/// Слой активации Sigmoid: `f(x) = 1 / (1 + e^-x)`
///
/// Классическая S-образная функция с выходом в диапазоне (0, 1).
/// Используется для бинарной классификации и gate-механизмов.
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

/// Слой активации ELU: `f(x) = x if x > 0 else alpha * (e^x - 1)`
///
/// Гладкая альтернатива ReLU с отрицательными значениями.
/// Помогает ускорить обучение и улучшить обобщение.
pub struct ELU {
    /// Параметр альфа для отрицательных значений (обычно 1.0)
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

/// Слой активации Softplus: `f(x) = log(1 + e^x)`
///
/// Гладкая аппроксимация ReLU. Всегда положительна.
/// Производная равна sigmoid(x).
pub struct Softplus {
    /// Параметр beta (по умолчанию 1.0)
    pub beta: f32,
    /// Порог для численной стабильности
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

/// Слой Softmax: `f(x)_i = e^x_i / sum(e^x_j)`
///
/// Преобразует вектор в распределение вероятностей.
/// Используется для многоклассовой классификации.
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