//! Fully connected (linear) layer with declarative shape/init API.

use crate::nn::init::Initializer;
use crate::nn::module::Module;
use crate::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Fully connected layer implementing `y = x @ W + b`.
///
/// In the graph-based architecture this layer does **not** store actual data.
/// Instead it owns symbolic `Tensor` handles that represent its weights and
/// biases as `Parameter` nodes in the ASG. The shapes and initializers are
/// registered in the [`GraphContext`]'s `ParameterRegistry` automatically.
///
/// # Example
///
/// ```ignore
/// use rustyasg::nn::{Linear, Module};
/// use rustyasg::tensor::GraphContext;
/// use std::rc::Rc;
/// use std::cell::RefCell;
///
/// let ctx = Rc::new(RefCell::new(GraphContext::new()));
/// let fc = Linear::new(&ctx, "fc1", 784, 128); // in=784, out=128
/// // ctx now knows: fc1.weights is [784, 128] Xavier, fc1.bias is [1, 128] Zeros.
/// ```
pub struct Linear {
    /// Symbolic handle for the weights tensor of shape `[in_features, out_features]`.
    pub weights: Tensor,
    /// Symbolic handle for the bias tensor of shape `[1, out_features]`
    /// (or `None` if the layer has no bias).
    pub bias: Option<Tensor>,
    /// Number of input features (columns of input tensor).
    pub in_features: usize,
    /// Number of output features (rows of weight tensor).
    pub out_features: usize,
}

impl Linear {
    /// Creates a fully connected layer with Xavier-uniform weights and zero bias.
    ///
    /// This is the recommended constructor. The layer's shapes are registered
    /// with the `GraphContext` automatically — no need to supply them to
    /// `ShapeInference` manually.
    ///
    /// # Arguments
    /// * `context` — the shared graph context.
    /// * `name` — a prefix for parameter names (e.g. `"fc1"` → `"fc1.weights"`, `"fc1.bias"`).
    /// * `in_features` — input dimensionality.
    /// * `out_features` — output dimensionality.
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        name: &str,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        Self::with_initializers(
            context,
            name,
            in_features,
            out_features,
            Initializer::XavierUniform,
            Some(Initializer::Zeros),
        )
    }

    /// Creates a layer without a bias term. Weights use Xavier-uniform init.
    pub fn without_bias(
        context: &Rc<RefCell<GraphContext>>,
        name: &str,
        in_features: usize,
        out_features: usize,
    ) -> Self {
        Self::with_initializers(
            context,
            name,
            in_features,
            out_features,
            Initializer::XavierUniform,
            None,
        )
    }

    /// Creates a layer with explicit initializers for weights and (optional) bias.
    ///
    /// # Arguments
    /// * `weight_init` — strategy for the weights tensor.
    /// * `bias_init` — `Some(init)` to include a bias, `None` for no bias.
    pub fn with_initializers(
        context: &Rc<RefCell<GraphContext>>,
        name: &str,
        in_features: usize,
        out_features: usize,
        weight_init: Initializer,
        bias_init: Option<Initializer>,
    ) -> Self {
        let weights_name = format!("{}.weights", name);
        let weights = Tensor::new_parameter_with_shape(
            context,
            &weights_name,
            vec![in_features, out_features],
            weight_init,
        );

        let bias = bias_init.map(|init| {
            let bias_name = format!("{}.bias", name);
            Tensor::new_parameter_with_shape(context, &bias_name, vec![1, out_features], init)
        });

        Self {
            weights,
            bias,
            in_features,
            out_features,
        }
    }
}

impl Module for Linear {
    /// Adds `x @ W + b` (or `x @ W` if no bias) to the graph.
    fn forward(&self, inputs: &Tensor) -> Tensor {
        let dot = inputs.dot(&self.weights);
        match &self.bias {
            Some(b) => &dot + b,
            None => dot,
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weights.clone()];
        if let Some(b) = &self.bias {
            params.push(b.clone());
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_registers_shapes() {
        let ctx = Rc::new(RefCell::new(GraphContext::new()));
        let _fc = Linear::new(&ctx, "fc1", 784, 128);

        let borrowed = ctx.borrow();
        let w_meta = borrowed.parameter_meta("fc1.weights").unwrap();
        assert_eq!(w_meta.shape, vec![784, 128]);

        let b_meta = borrowed.parameter_meta("fc1.bias").unwrap();
        assert_eq!(b_meta.shape, vec![1, 128]);
    }

    #[test]
    fn linear_without_bias_has_no_bias_param() {
        let ctx = Rc::new(RefCell::new(GraphContext::new()));
        let fc = Linear::without_bias(&ctx, "fc2", 32, 16);

        assert!(fc.bias.is_none());
        assert_eq!(fc.parameters().len(), 1);
        assert!(ctx.borrow().parameter_meta("fc2.bias").is_none());
    }
}
