//! Layer Normalization with declarative shape/init API.

use crate::asg::NodeType;
use crate::nn::init::Initializer;
use crate::nn::Module;
use crate::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Default epsilon for numerical stability.
const DEFAULT_EPS: f32 = 1e-5;

/// Layer normalization over the last axis.
///
/// Normalizes inputs as `y = gamma * (x - mean) / sqrt(var + eps) + beta`,
/// where `gamma` is initialized to ones and `beta` to zeros — the standard
/// PyTorch / TensorFlow defaults that preserve the original activations at
/// the start of training.
pub struct LayerNorm {
    /// Learnable scale of shape `[1, normalized_shape]`.
    pub gamma: Tensor,
    /// Learnable shift of shape `[1, normalized_shape]`.
    pub beta: Tensor,
    /// Epsilon for numerical stability.
    pub eps: f32,
    /// Size of the normalized axis.
    pub normalized_shape: usize,
}

impl LayerNorm {
    /// Creates a LayerNorm layer over `normalized_shape` features.
    ///
    /// # Arguments
    /// * `ctx` — shared graph context.
    /// * `name` — parameter name prefix (`"{name}.gamma"`, `"{name}.beta"`).
    /// * `normalized_shape` — size of the axis being normalized (typically `d_model`).
    pub fn new(ctx: &Rc<RefCell<GraphContext>>, name: &str, normalized_shape: usize) -> Self {
        Self::with_eps(ctx, name, normalized_shape, DEFAULT_EPS)
    }

    /// Creates a LayerNorm with a custom epsilon.
    pub fn with_eps(
        ctx: &Rc<RefCell<GraphContext>>,
        name: &str,
        normalized_shape: usize,
        eps: f32,
    ) -> Self {
        let gamma = Tensor::new_parameter_with_shape(
            ctx,
            &format!("{}.gamma", name),
            vec![1, normalized_shape],
            Initializer::Ones,
        );
        let beta = Tensor::new_parameter_with_shape(
            ctx,
            &format!("{}.beta", name),
            vec![1, normalized_shape],
            Initializer::Zeros,
        );
        Self {
            gamma,
            beta,
            eps,
            normalized_shape,
        }
    }
}

impl Module for LayerNorm {
    /// Forward pass using specialized `NodeType::LayerNorm` for correct autograd.
    fn forward(&self, x: &Tensor) -> Tensor {
        let ctx = &x.context;
        let node_id = ctx.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::LayerNorm {
                input: x.node_id,
                gamma: self.gamma.node_id,
                beta: self.beta.node_id,
                eps: self.eps,
            },
        );
        Tensor {
            node_id,
            context: Rc::clone(ctx),
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layernorm_registers_shapes() {
        let ctx = Rc::new(RefCell::new(GraphContext::new()));
        let _ln = LayerNorm::new(&ctx, "ln", 64);

        let borrowed = ctx.borrow();
        assert_eq!(
            borrowed.parameter_meta("ln.gamma").unwrap().shape,
            vec![1, 64]
        );
        assert_eq!(
            borrowed.parameter_meta("ln.beta").unwrap().shape,
            vec![1, 64]
        );
        // gamma initialized to ones, beta to zeros (standard LayerNorm init).
        assert_eq!(
            borrowed.parameter_meta("ln.gamma").unwrap().initializer,
            Initializer::Ones
        );
        assert_eq!(
            borrowed.parameter_meta("ln.beta").unwrap().initializer,
            Initializer::Zeros
        );
    }
}
