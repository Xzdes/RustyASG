//! # RustyASG: Graph-based Deep Learning Engine in Rust
//!
//! **RustyASG** is a modern experimental deep learning framework written in Rust.
//! Its key feature is an architecture built around the **Abstract Semantic Graph (ASG)**.
//!
//! ## Usage Example
//!
//! ```no_run
//! use std::rc::Rc;
//! use std::cell::RefCell;
//! use rustyasg::tensor::{GraphContext, Tensor};
//! use rustyasg::losses::mse_loss;
//!
//! // 1. Create graph context
//! let context = Rc::new(RefCell::new(GraphContext::new()));
//!
//! // 2. Define symbolic inputs
//! let input = Tensor::new_input(&context, "input");
//! let expected = Tensor::new_input(&context, "expected");
//!
//! // 3. Build computation graph
//! let prediction = input.relu(); // Just an example operation
//! let loss = mse_loss(&prediction, &expected);
//!
//! // Graph is ready for analysis, differentiation, and execution on a backend!
//! ```

// Declare public modules that constitute the core library API.
pub mod analysis;
pub mod asg;
pub mod autograd;
pub mod data;
pub mod losses;
pub mod metrics;
pub mod nn;
pub mod optimizers;
pub mod runtime;
pub mod serialization;
pub mod tensor;