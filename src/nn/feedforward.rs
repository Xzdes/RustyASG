//! Position-wise FeedForward network, a standard Transformer block component.
//!
//! Consists of two linear layers with a ReLU activation in between:
//! `output = Linear(ReLU(Linear(input)))`.

use crate::nn::{Linear, Module, ReLU};
use crate::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// FeedForward block: `embed_dim -> hidden_dim -> embed_dim`.
///
/// Parameter shapes are registered automatically via the underlying
/// [`Linear`] layers — no manual shape bookkeeping required.
pub struct FeedForward {
    linear1: Linear,
    relu: ReLU,
    linear2: Linear,
}

impl FeedForward {
    /// Creates a FeedForward network.
    ///
    /// # Arguments
    /// * `context` — shared graph context.
    /// * `name` — parameter-name prefix (child linears become `{name}.linear1` / `.linear2`).
    /// * `embed_dim` — input and output dimensionality.
    /// * `hidden_dim` — internal expansion size (typically `4 * embed_dim`).
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        name: &str,
        embed_dim: usize,
        hidden_dim: usize,
    ) -> Self {
        Self {
            linear1: Linear::new(context, &format!("{}.linear1", name), embed_dim, hidden_dim),
            relu: ReLU::new(),
            linear2: Linear::new(context, &format!("{}.linear2", name), hidden_dim, embed_dim),
        }
    }
}

impl Module for FeedForward {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        let x = self.linear1.forward(inputs);
        let x = self.relu.forward(&x);
        self.linear2.forward(&x)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.linear1.parameters());
        params.extend(self.linear2.parameters());
        params
    }
}
