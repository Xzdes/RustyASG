//! # Backend Trait Module
//!
//! This module defines the abstract interface ([`Backend`] trait) that all
//! execution backends must implement.
//!
//! ## Architecture
//!
//! RustyASG separates graph construction from execution through backends:
//!
//! ```text
//! Graph Construction (Tensor API) -> ASG -> Backend -> Results
//! ```
//!
//! ## Available Backends
//!
//! - [`CpuBackend`](super::cpu_backend::CpuBackend): Pure Rust CPU execution using ndarray
//! - [`WgpuBackend`](super::wgpu_backend::WgpuBackend): GPU execution using WebGPU (wgpu)
//!
//! ## Example Usage
//!
//! ```ignore
//! use rustyasg::runtime::backend::Backend;
//! use rustyasg::runtime::cpu_backend::CpuBackend;
//!
//! let backend = CpuBackend::new();
//!
//! // Load input data to device
//! let device_data = backend.load_data(&inputs)?;
//!
//! // Create initial memo with loaded data
//! let mut memo = HashMap::new();
//! for (name, tensor) in &tensors {
//!     memo.insert((graph.id, tensor.node_id), device_data[name].clone());
//! }
//!
//! // Execute graph
//! let (outputs, _memo) = backend.run(&graph, memo)?;
//!
//! // Retrieve results back to CPU
//! let results = backend.retrieve_data(&outputs)?;
//! ```

use crate::asg::{Asg, AsgId, NodeId, Value};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during graph execution (interpretation).
/// This error type is common to all backends.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum RuntimeError {
    #[error("Node with ID {0} not found in graph {1}. Verify that the graph was built correctly and all nodes exist.")]
    NodeNotFound(NodeId, AsgId),

    #[error("Graph with ID {0} not found in execution context. Ensure the graph was registered before execution.")]
    GraphNotFound(AsgId),

    #[error("Type mismatch: operation expected {expected}, but got {actual}. Check input data types.")]
    TypeError { expected: String, actual: String },

    #[error("Tensor shape error: {0}. Check input tensor dimensions.")]
    ShapeError(String),

    #[error("Missing value for input '{0}' (node ID: {1}). Add this value to initial_data when calling backend.run().")]
    MissingInput(String, NodeId),

    #[error("Missing value for parameter '{0}' (node ID: {1}). Initialize the parameter before executing the graph.")]
    MissingParameter(String, NodeId),

    #[error("Operation '{0}' is not implemented in current backend. Consider using an alternative operation or implement support.")]
    UnimplementedOperation(String),

    #[error("Computation error: {0}")]
    ComputationError(String),

    #[error("Memory error: {0}")]
    MemoryError(String),
}

/// Cache for storing already computed node values.
/// Key is (AsgId, NodeId).
pub type Memo<T> = HashMap<(AsgId, NodeId), T>;

/// Trait defining the common interface for execution environments (backends).
///
/// Any struct implementing this trait can take an ASG and data
/// and perform computations, returning the result.
pub trait Backend {
    /// Type representing device-specific data (CPU or GPU).
    /// For example, for GPU this could be a wrapper over wgpu::Buffer.
    type DeviceData: std::fmt::Debug;

    /// Prepares data for execution: allocates memory on the device
    /// and copies data from CPU.
    fn load_data(
        &self,
        data: &HashMap<String, Value>,
    ) -> Result<HashMap<String, Self::DeviceData>, RuntimeError>;

    /// Executes the graph, using and updating the computation cache.
    ///
    /// # Arguments
    /// * `main_asg` - The main graph to execute.
    /// * `initial_memo` - Cache with initial data (inputs, parameters) and possibly
    ///   results from previous computations (for linked graphs).
    ///
    /// # Returns
    /// A tuple containing:
    /// 1. Vector with the graph's output data.
    /// 2. Final state of the `Memo` cache with all intermediate results.
    fn run(
        &self,
        main_asg: &Asg,
        initial_memo: Memo<Self::DeviceData>,
    ) -> Result<(Vec<Self::DeviceData>, Memo<Self::DeviceData>), RuntimeError>;

    /// Retrieves results from the device back as CPU values (`Value`).
    fn retrieve_data(&self, device_data: &[Self::DeviceData]) -> Result<Vec<Value>, RuntimeError>;
}
