//! # Data Loading Module
//!
//! PyTorch-style Dataset and DataLoader API for efficient data handling.
//!
//! ## Key Components
//!
//! - [`Dataset`]: Trait for defining data sources
//! - [`DataLoader`]: Batched data iterator with shuffle support
//! - [`Sampler`]: Data sampling strategies (sequential, random)
//! - [`Transform`]: Data transformations (normalize, augmentation)
//!
//! ## Example
//!
//! ```ignore
//! use rustyasg::data::{Dataset, DataLoader, InMemoryDataset};
//!
//! // Create dataset
//! let dataset = InMemoryDataset::new(features, labels);
//!
//! // Create DataLoader
//! let loader = DataLoader::new(dataset, 32)
//!     .shuffle(true)
//!     .drop_last(false);
//!
//! // Iterate over batches
//! for (batch_x, batch_y) in loader.iter() {
//!     // Train on batch
//! }
//! ```
//!
//! ## Available Components
//!
//! ### Datasets
//! - [`InMemoryDataset`]: In-memory dataset with features and labels
//! - [`MapDataset`]: Dataset with transformation pipeline
//!
//! ### Samplers
//! - [`SequentialSampler`]: Iterate in order
//! - [`RandomSampler`]: Shuffle indices randomly
//! - [`BatchSampler`]: Group indices into batches
//!
//! ### Transforms
//! - [`Normalize`]: Normalize with mean and std
//! - [`RandomNoise`]: Add random noise for augmentation
//! - [`Compose`]: Chain multiple transforms

pub mod dataset;
pub mod dataloader;
pub mod sampler;
pub mod transforms;

pub use dataset::{Dataset, InMemoryDataset, MapDataset};
pub use dataloader::{DataLoader, Batch};
pub use sampler::{Sampler, SequentialSampler, RandomSampler, BatchSampler};
pub use transforms::{Transform, Compose, Normalize, RandomNoise};
