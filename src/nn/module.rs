//! Module defining the core `Module` trait for all neural network layers.

use crate::tensor::Tensor;

/// Trait defining the common interface for all layers/modules.
///
/// In the graph-based architecture, `Module` is any component that can
/// add a specific pattern of operations and parameters to the graph.
pub trait Module {
    /// Performs a "symbolic" forward pass, building the corresponding part of the graph.
    ///
    /// Takes an input symbolic tensor and returns an output symbolic tensor.
    fn forward(&self, inputs: &Tensor) -> Tensor;

    /// Returns a list of all trainable parameters (as symbolic tensors)
    /// that belong to this module.
    fn parameters(&self) -> Vec<Tensor>;
}