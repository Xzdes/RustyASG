// Clippy lints suppressed intentionally for the entire crate:
//
// - `too_many_arguments`: convolution-style APIs legitimately take many tuple
//   parameters (stride, padding, dilation, groups...). Wrapping them in
//   config structs hurts ergonomics more than it helps.
// - `type_complexity`: the backend trait returns
//   `(Vec<DeviceData>, Memo<DeviceData>)` which clippy flags, but hiding it
//   behind a type alias makes navigation harder.
// - `should_implement_trait`: internal `add` method on the gradient builder
//   has a different signature than `std::ops::Add::add` on purpose.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::should_implement_trait
)]

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
