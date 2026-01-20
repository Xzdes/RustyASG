//! # Optimizers Module
//!
//! This module provides gradient-based optimization algorithms for training
//! neural networks.
//!
//! ## Available Optimizers
//!
//! | Optimizer | Description | When to Use |
//! |-----------|-------------|-------------|
//! | [`Sgd`] | Classic SGD with momentum | Simple models, fine-tuning |
//! | [`Adam`] | Adaptive moment estimation | Most use cases (default choice) |
//! | [`AdamW`] | Adam with decoupled weight decay | Transformers, better regularization |
//! | [`RMSprop`] | RMS propagation | RNNs, non-stationary objectives |
//!
//! ## Learning Rate Schedulers
//!
//! | Scheduler | Description |
//! |-----------|-------------|
//! | [`StepLR`] | Decay LR every N epochs |
//! | [`ExponentialLR`] | Exponential decay |
//! | [`CosineAnnealingLR`] | Cosine annealing |
//! | [`LinearWarmupLR`] | Linear warmup phase |
//! | [`WarmupCosineAnnealingLR`] | Warmup + cosine annealing |
//!
//! ## Gradient Utilities
//!
//! - [`clip_grad_norm`]: Clip gradients by global L2 norm
//! - [`clip_grad_value`]: Clip gradients by absolute value
//!
//! ## Example
//!
//! ```ignore
//! use rustyasg::optimizers::{Adam, Optimizer, clip_grad_norm};
//!
//! let mut optimizer = Adam::new(0.001)
//!     .with_weight_decay(0.01);
//!
//! // Training loop
//! for epoch in 0..epochs {
//!     // Forward + backward pass (get gradients)
//!     clip_grad_norm(&mut gradients, 1.0);  // Prevent exploding gradients
//!     optimizer.step(&mut parameters, &gradients);
//! }
//! ```

use crate::asg::Value;
use ndarray::ArrayD;
use std::collections::HashMap;

/// Trait defining common interface for all optimizers.
pub trait Optimizer {
    /// Performs one optimization step, updating weights.
    fn step(&mut self, parameters: &mut HashMap<String, Value>, gradients: &HashMap<String, Value>);

    /// Resets accumulated gradients/state (for gradient accumulation).
    fn zero_grad(&mut self) {}

    /// Returns current learning rate.
    fn get_lr(&self) -> f32;

    /// Sets new learning rate.
    fn set_lr(&mut self, lr: f32);
}

// ============================================================================
// SGD - Stochastic Gradient Descent
// ============================================================================

/// SGD optimizer with optional momentum and weight decay.
///
/// Formula:
/// - Without momentum: `param = param - lr * grad`
/// - With momentum: `v = momentum * v + grad; param = param - lr * v`
/// - With weight decay: adds `lr * weight_decay * param`
pub struct Sgd {
    /// Learning rate
    pub lr: f32,
    /// Momentum coefficient (0 = no momentum)
    pub momentum: f32,
    /// Weight decay (L2 regularization)
    pub weight_decay: f32,
    /// Nesterov momentum
    pub nesterov: bool,
    /// Accumulated velocity for momentum
    velocity: HashMap<String, ArrayD<f32>>,
}

impl Sgd {
    /// Creates a basic SGD optimizer.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            velocity: HashMap::new(),
        }
    }

    /// Adds momentum.
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Adds weight decay.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Enables Nesterov momentum.
    pub fn with_nesterov(mut self) -> Self {
        self.nesterov = true;
        self
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, parameters: &mut HashMap<String, Value>, gradients: &HashMap<String, Value>) {
        for (param_name, grad_value) in gradients {
            if let (Some(Value::Tensor(param_value)), Value::Tensor(grad_tensor)) =
                (parameters.get_mut(param_name), grad_value)
            {
                let mut grad = grad_tensor.clone();

                // Weight decay
                if self.weight_decay > 0.0 {
                    grad = &grad + &(self.weight_decay * &*param_value);
                }

                if self.momentum > 0.0 {
                    // Momentum update
                    let v = self.velocity
                        .entry(param_name.clone())
                        .or_insert_with(|| ArrayD::zeros(param_value.shape()));

                    *v = self.momentum * &*v + &grad;

                    if self.nesterov {
                        grad = &grad + &(self.momentum * &*v);
                    } else {
                        grad = v.clone();
                    }
                }

                // param = param - lr * grad
                ndarray::azip!((p in param_value, &g in &grad) *p = *p - self.lr * g);
            }
        }
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

// ============================================================================
// Adam - Adaptive Moment Estimation
// ============================================================================

/// Adam optimizer.
///
/// Adaptive optimizer combining momentum and RMSprop.
/// Formula:
/// ```text
/// m = beta1 * m + (1 - beta1) * grad
/// v = beta2 * v + (1 - beta2) * grad^2
/// m_hat = m / (1 - beta1^t)
/// v_hat = v / (1 - beta2^t)
/// param = param - lr * m_hat / (sqrt(v_hat) + eps)
/// ```
pub struct Adam {
    /// Learning rate
    pub lr: f32,
    /// Exponential averaging coefficient for first moment
    pub beta1: f32,
    /// Exponential averaging coefficient for second moment
    pub beta2: f32,
    /// Small constant for numerical stability
    pub eps: f32,
    /// Weight decay (L2 regularization, applied to gradient)
    pub weight_decay: f32,
    /// First moments (m)
    m: HashMap<String, ArrayD<f32>>,
    /// Second moments (v)
    v: HashMap<String, ArrayD<f32>>,
    /// Step counter for bias correction
    t: usize,
}

impl Adam {
    /// Creates Adam with default parameters.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    /// Sets beta1.
    pub fn with_beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Sets beta2.
    pub fn with_beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Sets epsilon.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    /// Adds weight decay.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
}

impl Optimizer for Adam {
    fn step(&mut self, parameters: &mut HashMap<String, Value>, gradients: &HashMap<String, Value>) {
        self.t += 1;
        let t = self.t as f32;

        // Bias correction factors
        let bias_correction1 = 1.0 - self.beta1.powf(t);
        let bias_correction2 = 1.0 - self.beta2.powf(t);

        for (param_name, grad_value) in gradients {
            if let (Some(Value::Tensor(param_value)), Value::Tensor(grad_tensor)) =
                (parameters.get_mut(param_name), grad_value)
            {
                let mut grad = grad_tensor.clone();

                // Weight decay (added to gradient - L2 regularization)
                if self.weight_decay > 0.0 {
                    grad = &grad + &(self.weight_decay * &*param_value);
                }

                // Initialize or get m and v
                let m = self.m
                    .entry(param_name.clone())
                    .or_insert_with(|| ArrayD::zeros(param_value.shape()));
                let v = self.v
                    .entry(param_name.clone())
                    .or_insert_with(|| ArrayD::zeros(param_value.shape()));

                // m = beta1 * m + (1 - beta1) * grad
                *m = self.beta1 * &*m + (1.0 - self.beta1) * &grad;

                // v = beta2 * v + (1 - beta2) * grad^2
                *v = self.beta2 * &*v + (1.0 - self.beta2) * &(&grad * &grad);

                // Bias-corrected estimates
                let m_hat = &*m / bias_correction1;
                let v_hat = &*v / bias_correction2;

                // param = param - lr * m_hat / (sqrt(v_hat) + eps)
                ndarray::azip!((p in param_value, &mh in &m_hat, &vh in &v_hat) {
                    *p = *p - self.lr * mh / (vh.sqrt() + self.eps);
                });
            }
        }
    }

    fn zero_grad(&mut self) {
        // Adam stores state, zero_grad doesn't clear it
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

// ============================================================================
// AdamW - Adam with Decoupled Weight Decay
// ============================================================================

/// AdamW optimizer.
///
/// Differs from Adam in that weight decay is applied directly to weights,
/// not to the gradient. This leads to better regularization.
///
/// Formula differs: `param = param - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * param)`
pub struct AdamW {
    /// Learning rate
    pub lr: f32,
    /// Beta1 coefficient
    pub beta1: f32,
    /// Beta2 coefficient
    pub beta2: f32,
    /// Epsilon for stability
    pub eps: f32,
    /// Weight decay (decoupled)
    pub weight_decay: f32,
    /// First moments
    m: HashMap<String, ArrayD<f32>>,
    /// Second moments
    v: HashMap<String, ArrayD<f32>>,
    /// Step counter
    t: usize,
}

impl AdamW {
    /// Creates AdamW with default parameters.
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01, // Typical value for AdamW
            m: HashMap::new(),
            v: HashMap::new(),
            t: 0,
        }
    }

    /// Sets weight decay.
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Sets beta1.
    pub fn with_beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Sets beta2.
    pub fn with_beta2(mut self, beta2: f32) -> Self {
        self.beta2 = beta2;
        self
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, parameters: &mut HashMap<String, Value>, gradients: &HashMap<String, Value>) {
        self.t += 1;
        let t = self.t as f32;

        let bias_correction1 = 1.0 - self.beta1.powf(t);
        let bias_correction2 = 1.0 - self.beta2.powf(t);

        for (param_name, grad_value) in gradients {
            if let (Some(Value::Tensor(param_value)), Value::Tensor(grad_tensor)) =
                (parameters.get_mut(param_name), grad_value)
            {
                let grad = grad_tensor.clone();

                // Decoupled weight decay: first apply to weights
                if self.weight_decay > 0.0 {
                    ndarray::azip!((p in &mut *param_value) {
                        *p = *p * (1.0 - self.lr * self.weight_decay);
                    });
                }

                // Standard Adam update
                let m = self.m
                    .entry(param_name.clone())
                    .or_insert_with(|| ArrayD::zeros(param_value.shape()));
                let v = self.v
                    .entry(param_name.clone())
                    .or_insert_with(|| ArrayD::zeros(param_value.shape()));

                *m = self.beta1 * &*m + (1.0 - self.beta1) * &grad;
                *v = self.beta2 * &*v + (1.0 - self.beta2) * &(&grad * &grad);

                let m_hat = &*m / bias_correction1;
                let v_hat = &*v / bias_correction2;

                ndarray::azip!((p in param_value, &mh in &m_hat, &vh in &v_hat) {
                    *p = *p - self.lr * mh / (vh.sqrt() + self.eps);
                });
            }
        }
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

// ============================================================================
// RMSprop
// ============================================================================

/// RMSprop optimizer.
///
/// Adaptive optimizer using running average of squared gradients.
pub struct RMSprop {
    pub lr: f32,
    pub alpha: f32,     // Smoothing constant (default 0.99)
    pub eps: f32,
    pub weight_decay: f32,
    pub momentum: f32,
    v: HashMap<String, ArrayD<f32>>,
    buf: HashMap<String, ArrayD<f32>>,
}

impl RMSprop {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            v: HashMap::new(),
            buf: HashMap::new(),
        }
    }

    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self, parameters: &mut HashMap<String, Value>, gradients: &HashMap<String, Value>) {
        // Copy parameters to avoid borrow checker issues
        let lr = self.lr;
        let alpha = self.alpha;
        let eps = self.eps;
        let weight_decay = self.weight_decay;
        let momentum = self.momentum;

        for (param_name, grad_value) in gradients {
            if let (Some(Value::Tensor(param_value)), Value::Tensor(grad_tensor)) =
                (parameters.get_mut(param_name), grad_value)
            {
                let mut grad = grad_tensor.clone();

                if weight_decay > 0.0 {
                    grad = &grad + &(weight_decay * &*param_value);
                }

                let v = self.v
                    .entry(param_name.clone())
                    .or_insert_with(|| ArrayD::zeros(param_value.shape()));

                // v = alpha * v + (1 - alpha) * grad^2
                *v = alpha * &*v + (1.0 - alpha) * &(&grad * &grad);

                if momentum > 0.0 {
                    let buf = self.buf
                        .entry(param_name.clone())
                        .or_insert_with(|| ArrayD::zeros(param_value.shape()));

                    // buf = momentum * buf + grad / sqrt(v + eps)
                    ndarray::azip!((b in &mut *buf, &g in &grad, &vi in &*v) {
                        *b = momentum * *b + g / (vi.sqrt() + eps);
                    });

                    ndarray::azip!((p in param_value, &b in &*buf) {
                        *p = *p - lr * b;
                    });
                } else {
                    ndarray::azip!((p in param_value, &g in &grad, &vi in &*v) {
                        *p = *p - lr * g / (vi.sqrt() + eps);
                    });
                }
            }
        }
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }
}

// ============================================================================
// Learning Rate Schedulers
// ============================================================================

/// Trait for Learning Rate Schedulers.
pub trait LRScheduler {
    /// Returns learning rate for given epoch/step.
    fn get_lr(&self, epoch: usize, step: usize) -> f32;
}

/// Step LR - decreases LR every `step_size` epochs.
pub struct StepLR {
    pub initial_lr: f32,
    pub step_size: usize,
    pub gamma: f32,
}

impl StepLR {
    pub fn new(initial_lr: f32, step_size: usize, gamma: f32) -> Self {
        Self { initial_lr, step_size, gamma }
    }
}

impl LRScheduler for StepLR {
    fn get_lr(&self, epoch: usize, _step: usize) -> f32 {
        self.initial_lr * self.gamma.powi((epoch / self.step_size) as i32)
    }
}

/// Exponential LR - exponential decay.
pub struct ExponentialLR {
    pub initial_lr: f32,
    pub gamma: f32,
}

impl ExponentialLR {
    pub fn new(initial_lr: f32, gamma: f32) -> Self {
        Self { initial_lr, gamma }
    }
}

impl LRScheduler for ExponentialLR {
    fn get_lr(&self, epoch: usize, _step: usize) -> f32 {
        self.initial_lr * self.gamma.powi(epoch as i32)
    }
}

/// Cosine Annealing LR - cosine decay.
pub struct CosineAnnealingLR {
    pub initial_lr: f32,
    pub min_lr: f32,
    pub total_epochs: usize,
}

impl CosineAnnealingLR {
    pub fn new(initial_lr: f32, total_epochs: usize) -> Self {
        Self {
            initial_lr,
            min_lr: 0.0,
            total_epochs,
        }
    }

    pub fn with_min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn get_lr(&self, epoch: usize, _step: usize) -> f32 {
        let progress = epoch as f32 / self.total_epochs as f32;
        self.min_lr + (self.initial_lr - self.min_lr) * (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0
    }
}

/// Linear Warmup LR - linear warmup.
pub struct LinearWarmupLR {
    pub target_lr: f32,
    pub warmup_steps: usize,
}

impl LinearWarmupLR {
    pub fn new(target_lr: f32, warmup_steps: usize) -> Self {
        Self { target_lr, warmup_steps }
    }
}

impl LRScheduler for LinearWarmupLR {
    fn get_lr(&self, _epoch: usize, step: usize) -> f32 {
        if step >= self.warmup_steps {
            self.target_lr
        } else {
            self.target_lr * (step as f32 / self.warmup_steps as f32)
        }
    }
}

/// Warmup + Cosine Annealing - combined scheduler.
pub struct WarmupCosineAnnealingLR {
    pub target_lr: f32,
    pub min_lr: f32,
    pub warmup_steps: usize,
    pub total_steps: usize,
}

impl WarmupCosineAnnealingLR {
    pub fn new(target_lr: f32, warmup_steps: usize, total_steps: usize) -> Self {
        Self {
            target_lr,
            min_lr: 0.0,
            warmup_steps,
            total_steps,
        }
    }

    pub fn with_min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }
}

impl LRScheduler for WarmupCosineAnnealingLR {
    fn get_lr(&self, _epoch: usize, step: usize) -> f32 {
        if step < self.warmup_steps {
            // Linear warmup
            self.target_lr * (step as f32 / self.warmup_steps as f32)
        } else {
            // Cosine annealing
            let progress = (step - self.warmup_steps) as f32 / (self.total_steps - self.warmup_steps) as f32;
            self.min_lr + (self.target_lr - self.min_lr) * (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0
        }
    }
}

// ============================================================================
// Gradient Clipping Utilities
// ============================================================================

/// Clips gradients by norm.
pub fn clip_grad_norm(gradients: &mut HashMap<String, Value>, max_norm: f32) -> f32 {
    // Compute total norm
    let mut total_norm_sq = 0.0f32;
    for grad_value in gradients.values() {
        if let Value::Tensor(grad) = grad_value {
            total_norm_sq += grad.iter().map(|x| x * x).sum::<f32>();
        }
    }
    let total_norm = total_norm_sq.sqrt();

    // If norm exceeds max_norm, scale gradients
    if total_norm > max_norm {
        let clip_coef = max_norm / (total_norm + 1e-6);
        for grad_value in gradients.values_mut() {
            if let Value::Tensor(grad) = grad_value {
                grad.mapv_inplace(|x| x * clip_coef);
            }
        }
    }

    total_norm
}

/// Clips gradients by value.
pub fn clip_grad_value(gradients: &mut HashMap<String, Value>, max_value: f32) {
    for grad_value in gradients.values_mut() {
        if let Value::Tensor(grad) = grad_value {
            grad.mapv_inplace(|x| x.clamp(-max_value, max_value));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_sgd_basic() {
        let mut sgd = Sgd::new(0.1);
        let mut params = HashMap::new();
        params.insert("w".to_string(), Value::Tensor(array![1.0, 2.0, 3.0].into_dyn()));

        let mut grads = HashMap::new();
        grads.insert("w".to_string(), Value::Tensor(array![0.1, 0.2, 0.3].into_dyn()));

        sgd.step(&mut params, &grads);

        if let Value::Tensor(w) = &params["w"] {
            assert!((w[0] - 0.99).abs() < 1e-5);
            assert!((w[1] - 1.98).abs() < 1e-5);
        }
    }

    #[test]
    fn test_adam_basic() {
        let mut adam = Adam::new(0.001);
        let mut params = HashMap::new();
        params.insert("w".to_string(), Value::Tensor(array![1.0, 2.0].into_dyn()));

        let grads = HashMap::from([
            ("w".to_string(), Value::Tensor(array![0.1, 0.2].into_dyn()))
        ]);

        // Several steps
        for _ in 0..10 {
            adam.step(&mut params, &grads);
        }

        // Parameters should change
        if let Value::Tensor(w) = &params["w"] {
            assert!(w[0] < 1.0);
            assert!(w[1] < 2.0);
        }
    }

    #[test]
    fn test_cosine_annealing() {
        let scheduler = CosineAnnealingLR::new(0.1, 100);

        assert!((scheduler.get_lr(0, 0) - 0.1).abs() < 1e-5);
        assert!((scheduler.get_lr(50, 0) - 0.05).abs() < 1e-5);
        assert!((scheduler.get_lr(100, 0) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_gradient_clipping() {
        let mut grads = HashMap::new();
        grads.insert("w".to_string(), Value::Tensor(array![3.0, 4.0].into_dyn()));

        let norm = clip_grad_norm(&mut grads, 1.0);
        assert!((norm - 5.0).abs() < 1e-5); // Original norm is 5

        if let Value::Tensor(g) = &grads["w"] {
            // After clipping norm should be 1.0
            let new_norm = (g[0] * g[0] + g[1] * g[1]).sqrt();
            assert!((new_norm - 1.0).abs() < 1e-5);
        }
    }
}
