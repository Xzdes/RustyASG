// --- File: src/serialization/mod.rs ---

//! Model serialization and deserialization.
//!
//! Supported formats:
//! - **SafeTensors**: safe binary format for tensor storage.
//! - **JSON**: human-readable format for metadata and configuration.
//! - **Checkpoint**: full snapshots of model and optimizer state.
//!
//! # Examples
//!
//! ```rust,ignore
//! use rustyasg::serialization::{save_safetensors, load_safetensors};
//! use std::collections::HashMap;
//!
//! // Save weights.
//! let mut weights = HashMap::new();
//! weights.insert("layer1.weight".to_string(), tensor_data);
//! save_safetensors("model.safetensors", &weights)?;
//!
//! // Load weights.
//! let loaded = load_safetensors("model.safetensors")?;
//! ```

pub mod checkpoint;
pub mod safetensors_io;

pub use checkpoint::{load_checkpoint, save_checkpoint, Checkpoint, CheckpointConfig};
pub use safetensors_io::{load_safetensors, save_safetensors, SafeTensorsError};
