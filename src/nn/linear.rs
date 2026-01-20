//! Module implementing the fully connected (linear) layer in the graph paradigm.

use crate::nn::module::Module;
use crate::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Fully connected (linear) layer.
///
/// In the graph-based architecture, this layer does not store actual data. Instead,
/// it owns symbolic `Tensor` handles that represent its weights and biases
/// as `Parameter` nodes in the ASG.
///
/// The `forward` method adds operations to the graph corresponding to the formula `y = xW + b`.
pub struct Linear {
    /// Symbolic handle for the weights tensor.
    pub weights: Tensor,
    /// Symbolic handle for the bias tensor.
    pub bias: Tensor,
}

impl Linear {
    /// Creates a new fully connected layer, registering its parameters in the graph.
    ///
    /// # Arguments
    ///
    /// * `context` - Reference to the `GraphContext` where the graph is being built.
    /// * `name` - Base name for this layer so that parameters have unique names
    ///   in the graph (e.g., "layer1.weights", "layer1.bias").
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        name: &str,
    ) -> Self {
        // Create symbolic parameter nodes in the graph.
        // Actual values and shapes for these parameters will be provided
        // later, before execution and shape inference.
        let weights_name = format!("{}.weights", name);
        let bias_name = format!("{}.bias", name);

        let weights = Tensor::new_parameter(context, &weights_name);
        let bias = Tensor::new_parameter(context, &bias_name);

        Self { weights, bias }
    }
}

impl Module for Linear {
    /// Adds operations to the graph for the forward pass through the linear layer.
    ///
    /// Constructs a subgraph corresponding to `inputs.dot(weights) + bias`.
    fn forward(&self, inputs: &Tensor) -> Tensor {
        let dot_product = inputs.dot(&self.weights);
        let final_output = &dot_product + &self.bias;
        final_output
    }

    /// Returns a list of symbolic handles for the layer's trainable parameters.
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}
