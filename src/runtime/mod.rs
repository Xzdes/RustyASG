//! # Runtime Execution Backends
//!
//! This module contains execution backends that run computation graphs.
//!
//! ## Available Backends
//!
//! | Backend | Module | Hardware | Async |
//! |---------|--------|----------|-------|
//! | CPU | [`cpu_backend`] | Any CPU | No |
//! | GPU | [`wgpu_backend`] | Vulkan/Metal/DX12/WebGPU | Yes |
//!
//! ## Backend Selection
//!
//! - **CPU**: Best for debugging, small models, or when GPU isn't available
//! - **GPU**: Best for large models, batched inference, training
//!
//! ## Architecture
//!
//! All backends implement the [`Backend`](backend::Backend) trait:
//!
//! ```text
//! Backend trait:
//! ├── load_data()    - Transfer data to device memory
//! ├── run()          - Execute computation graph
//! └── retrieve_data() - Transfer results back to CPU
//! ```
//!
//! See [`backend`] module for trait documentation.

pub mod backend;
pub mod cpu_backend;
pub mod wgpu_backend;