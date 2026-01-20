//  src/nn/norm.rs
//! Layer Normalization for graph-based architecture with correct autograd.

use crate::asg::NodeType;
use crate::nn::Module;
use crate::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Small constant for numerical stability.
const EPS: f32 = 1e-5;

/// Layer normalization layer that normalizes along the last axis.
pub struct LayerNorm {
    pub gamma: Tensor, // learnable scale
    pub beta: Tensor,  // learnable shift
    pub eps: f32,      // epsilon constant
}

impl LayerNorm {
    /// Creates a new layer, registering `gamma` and `beta` parameters in the graph.
    pub fn new(ctx: &Rc<RefCell<GraphContext>>, name: &str) -> Self {
        let gamma_name = format!("{}.gamma", name);
        let beta_name = format!("{}.beta", name);
        let gamma = Tensor::new_parameter(ctx, &gamma_name);
        let beta = Tensor::new_parameter(ctx, &beta_name);
        Self {
            gamma,
            beta,
            eps: EPS,
        }
    }

    /// Creates a layer with custom epsilon.
    pub fn with_eps(ctx: &Rc<RefCell<GraphContext>>, name: &str, eps: f32) -> Self {
        let mut ln = Self::new(ctx, name);
        ln.eps = eps;
        ln
    }
}

impl Module for LayerNorm {
    /// Forward pass:  y = gamma * (x - mean) / sqrt(var + eps) + beta
    /// Uses specialized NodeType::LayerNorm for correct autograd.
    fn forward(&self, x: &Tensor) -> Tensor {
        // Use context from input tensor x
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
