//! BatchNormalization with declarative shape/init API.
//!
//! Implements Batch Normalization with train/eval modes, running statistics
//! (stub), and learnable gamma/beta parameters.

use crate::nn::init::Initializer;
use crate::nn::Module;
use crate::tensor::{GraphContext, Tensor};
use ndarray::arr0;
use std::cell::RefCell;
use std::rc::Rc;

/// Small constant for numerical stability.
const EPS: f32 = 1e-5;

/// Default momentum for running-statistics EMA.
const DEFAULT_MOMENTUM: f32 = 0.1;

/// Batch Normalization layer.
///
/// Normalizes inputs along the batch dimension using
/// `y = gamma * (x - mean) / sqrt(var + eps) + beta`.
///
/// # Limitations (Phase 2)
/// The current forward pass always uses batch statistics — running statistics
/// for inference mode are not yet materialized through the ASG. See ROADMAP
/// for the full BatchNorm work item.
pub struct BatchNorm {
    /// Learnable scale of shape `[num_features]`.
    pub gamma: Tensor,
    /// Learnable shift of shape `[num_features]`.
    pub beta: Tensor,
    /// Epsilon literal (embedded as a constant in the graph).
    eps: Tensor,
    /// EMA momentum for running statistics.
    pub momentum: f32,
    /// Training / inference mode flag.
    pub training: bool,
    /// Layer name (used for running-stats lookup).
    pub name: String,
    /// Number of features (size of the normalized channel axis).
    pub num_features: usize,
}

impl BatchNorm {
    /// Creates a BatchNorm layer over `num_features` channels.
    ///
    /// # Arguments
    /// * `ctx` — shared graph context.
    /// * `name` — unique layer name (used as parameter-name prefix and for running stats).
    /// * `num_features` — channel axis size.
    pub fn new(ctx: &Rc<RefCell<GraphContext>>, name: &str, num_features: usize) -> Self {
        let gamma = Tensor::new_parameter_with_shape(
            ctx,
            &format!("{}.gamma", name),
            vec![num_features],
            Initializer::Ones,
        );
        let beta = Tensor::new_parameter_with_shape(
            ctx,
            &format!("{}.beta", name),
            vec![num_features],
            Initializer::Zeros,
        );
        let eps = Tensor::new_literal(ctx, arr0(EPS).into_dyn(), &format!("{}.eps", name));

        Self {
            gamma,
            beta,
            eps,
            momentum: DEFAULT_MOMENTUM,
            training: true,
            name: name.to_string(),
            num_features,
        }
    }

    /// Sets the EMA momentum for running statistics.
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }
}

impl Module for BatchNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mean = x.mean();
        let x_minus_mean = x - &mean;

        let squared = &x_minus_mean * &x_minus_mean;
        let variance = squared.mean();

        let std = (&variance + &self.eps).sqrt();
        let normalized = &x_minus_mean / &std;

        let scaled = &normalized * &self.gamma;
        &scaled + &self.beta
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batchnorm_registers_shapes() {
        let ctx = Rc::new(RefCell::new(GraphContext::new()));
        let _bn = BatchNorm::new(&ctx, "bn1", 32);

        let borrowed = ctx.borrow();
        assert_eq!(
            borrowed.parameter_meta("bn1.gamma").unwrap().shape,
            vec![32]
        );
        assert_eq!(borrowed.parameter_meta("bn1.beta").unwrap().shape, vec![32]);
    }

    #[test]
    fn batchnorm_train_eval_toggle() {
        let ctx = Rc::new(RefCell::new(GraphContext::new()));
        let mut bn = BatchNorm::new(&ctx, "bn1", 16);

        bn.eval();
        assert!(!bn.training);
        bn.train();
        assert!(bn.training);
    }
}
