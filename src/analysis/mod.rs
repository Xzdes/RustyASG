//! # Graph Analysis Module
//!
//! This module contains analysis passes that process the ASG before execution.
//!
//! ## Available Passes
//!
//! - [`ShapeInference`](shape_inference::ShapeInference): Propagates tensor shapes
//!   through the graph, detecting shape mismatches before runtime.
//!
//! ## How It Works
//!
//! Analysis passes traverse the graph and compute/validate metadata:
//!
//! ```text
//! ASG (untyped) -> Shape Inference -> ASG (with shapes/dtypes)
//! ```
//!
//! This enables:
//! - **Early error detection**: Shape mismatches caught before execution
//! - **Memory planning**: Know buffer sizes ahead of time
//! - **Optimization opportunities**: Shape-aware graph transformations
//!
//! ## Example
//!
//! ```ignore
//! use rustyasg::analysis::shape_inference::ShapeInference;
//!
//! let mut graph = context.borrow().main_graph().clone();
//!
//! // Provide initial shapes for inputs
//! let shapes = HashMap::from([
//!     ("input".to_string(), (vec![32, 784], DType::F32)),
//! ]);
//!
//! // Run shape inference
//! ShapeInference::run(&mut graph, &shapes)?;
//!
//! // Now all nodes have shape information
//! ```

pub mod shape_inference;