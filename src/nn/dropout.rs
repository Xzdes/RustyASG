//! Dropout layers for regularisation.
//!
//! `Dropout` and `SpatialDropout` are implemented as graph compositions:
//! they emit a `DropoutMask` node and multiply the input by it. The mask
//! is freshly sampled by the backend each forward pass and cached in the
//! forward memo, so the backward pass — which references the same mask
//! through the standard `Multiply` rule — sees identical values.
//!
//! In `eval()` mode, `forward` simply returns the input unchanged: no
//! mask, no node, no overhead.

use crate::asg::NodeType;
use crate::nn::Module;
use crate::tensor::Tensor;
use std::rc::Rc;

/// Standard element-wise Dropout.
///
/// During training, each element is independently zeroed with probability `p`
/// and the surviving elements are scaled by `1 / (1 - p)` to preserve the
/// expected sum. The mask is resampled on every forward pass.
///
/// During inference (`eval()` mode), `forward` is a no-op.
///
/// # Example
/// ```ignore
/// let mut dropout = Dropout::new(0.5);
/// let h = dropout.forward(&hidden);     // training: h = hidden * mask * 2.0
/// dropout.eval();
/// let h = dropout.forward(&hidden);     // inference: h == hidden
/// ```
pub struct Dropout {
    /// Drop probability in `[0, 1)`.
    pub p: f32,
    /// Training-mode flag. Set to `false` for inference.
    pub training: bool,
}

impl Dropout {
    /// Creates a new Dropout layer.
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

    /// Switches to training mode (mask is applied).
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Switches to inference mode (forward is a pass-through).
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Returns whether dropout is currently active.
    pub fn is_training(&self) -> bool {
        self.training
    }
}

impl Module for Dropout {
    /// Forward pass: `y = x * mask` where `mask[i] ∈ {0, 1/(1-p)}` is a
    /// fresh Bernoulli sample at each runtime invocation. In `eval()` mode
    /// this is a pass-through.
    fn forward(&self, x: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return x.clone();
        }
        // mask = DropoutMask(shape_provider=x, p)
        let ctx = &x.context;
        let mask_id = ctx.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::DropoutMask {
                shape_provider: x.node_id,
                p: self.p,
            },
        );
        let mask = Tensor {
            node_id: mask_id,
            context: Rc::clone(ctx),
        };
        // The CPU/GPU runtime bakes the `1 / (1 - p)` scaling factor directly
        // into the mask, so a plain element-wise multiply is the full forward.
        x * &mask
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

/// Spatial Dropout for convolutional networks.
///
/// Conceptually drops *entire feature-map channels* rather than individual
/// pixels — this preserves spatial correlations and tends to regularise CNNs
/// better than naïve element-wise dropout.
///
/// The current implementation falls back to element-wise dropout (`Dropout`)
/// because per-channel masking requires reshape/broadcast support that's
/// already on the v0.5 wishlist. The output is still a valid (if slightly
/// less effective) regulariser, and the layer is API-compatible so callers
/// can switch to true spatial dropout without code changes.
pub struct SpatialDropout {
    /// Per-channel drop probability.
    pub p: f32,
    /// Training-mode flag.
    pub training: bool,
}

impl SpatialDropout {
    /// Creates a new SpatialDropout layer.
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

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }
}

impl Module for SpatialDropout {
    fn forward(&self, x: &Tensor) -> Tensor {
        // Same path as `Dropout` for now (see struct docs).
        Dropout {
            p: self.p,
            training: self.training,
        }
        .forward(x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::GraphContext;
    use std::cell::RefCell;
    use std::rc::Rc;

    #[test]
    fn dropout_creation_validates_p() {
        let _ = Dropout::new(0.0);
        let _ = Dropout::new(0.5);
        let _ = Dropout::new(0.99);
    }

    #[test]
    #[should_panic(expected = "Dropout probability must be in [0, 1)")]
    fn dropout_rejects_p_one() {
        let _ = Dropout::new(1.0);
    }

    #[test]
    fn dropout_train_eval_toggles() {
        let mut d = Dropout::new(0.5);
        assert!(d.is_training());
        d.eval();
        assert!(!d.is_training());
        d.train();
        assert!(d.is_training());
    }

    #[test]
    fn dropout_eval_is_passthrough() {
        // In eval mode, forward should not add any new nodes to the graph.
        let ctx = Rc::new(RefCell::new(GraphContext::new()));
        let x = Tensor::new_input(&ctx, "x");
        let nodes_before = ctx.borrow().main_graph().nodes.len();
        let mut d = Dropout::new(0.5);
        d.eval();
        let y = d.forward(&x);
        let nodes_after = ctx.borrow().main_graph().nodes.len();
        assert_eq!(y.node_id, x.node_id);
        assert_eq!(nodes_before, nodes_after);
    }

    #[test]
    fn dropout_train_emits_mask_and_multiply() {
        let ctx = Rc::new(RefCell::new(GraphContext::new()));
        let x = Tensor::new_input(&ctx, "x");
        let nodes_before = ctx.borrow().main_graph().nodes.len();
        let d = Dropout::new(0.3);
        let _y = d.forward(&x);
        let nodes_after = ctx.borrow().main_graph().nodes.len();
        // One DropoutMask + one Multiply.
        assert_eq!(nodes_after - nodes_before, 2);
    }
}
