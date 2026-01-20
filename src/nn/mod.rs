//! # Neural Network Layers Module
//!
//! This module contains building blocks for constructing neural networks.
//!
//! In the graph-based architecture, each "layer" is a constructor that adds
//! a specific pattern of nodes (operations and parameters) to the ASG.
//!
//! ## Available Layers
//!
//! ### Core Layers
//! - [`Linear`]: Fully connected / dense layer
//! - [`Conv2d`]: 2D convolution with configurable stride, padding, dilation
//! - [`ConvTranspose2d`]: Transposed 2D convolution (deconvolution)
//! - [`Embedding`]: Embedding lookup table
//!
//! ### Normalization
//! - [`LayerNorm`]: Layer normalization
//! - [`BatchNorm`]: Batch normalization
//!
//! ### Activations
//! - [`ReLU`], [`LeakyReLU`], [`ELU`]: Rectified linear units
//! - [`Sigmoid`], [`Tanh`]: Classic activations
//! - [`GELU`], [`SiLU`]/[`Swish`]: Modern smooth activations
//! - [`Softmax`], [`Softplus`]: Output activations
//!
//! ### Attention & Transformers
//! - [`MultiHeadAttention`]: Multi-head self/cross attention
//! - [`TransformerBlock`]: Complete transformer encoder block
//! - [`FeedForward`]: Position-wise feed-forward network
//!
//! ### Positional Encodings
//! - [`SinusoidalPositionalEncoding`]: Fixed sinusoidal encoding
//! - [`LearnedPositionalEmbedding`]: Learnable position embeddings
//! - [`RotaryPositionEmbedding`]: RoPE (used in LLaMA, etc.)
//! - [`ALiBi`]: Attention with Linear Biases
//!
//! ### Pooling
//! - [`MaxPool2d`], [`AvgPool2d`]: Standard pooling
//! - [`GlobalAvgPool2d`], [`AdaptiveAvgPool2d`]: Global/adaptive pooling
//!
//! ### Regularization
//! - [`Dropout`]: Standard dropout
//! - [`SpatialDropout`]: Dropout for convolutional features
//!
//! ## Example
//!
//! ```ignore
//! use rustyasg::nn::{Linear, ReLU, Module};
//! use rustyasg::tensor::{GraphContext, Tensor};
//!
//! let ctx = Rc::new(RefCell::new(GraphContext::new()));
//! let linear = Linear::new(&ctx, "fc1", 784, 128);
//! let relu = ReLU;
//!
//! let x = Tensor::new_input(&ctx, "input");
//! let h = relu.forward(&linear.forward(&x));
//! ```

// Declare all submodules
pub mod activations;
pub mod attention;
pub mod batchnorm;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod feedforward;
pub mod linear;
pub mod module;
pub mod norm;
pub mod pooling;
pub mod positional;
pub mod transformer;

// Re-export structures for convenience

// Activations
pub use activations::{ELU, GELU, LeakyReLU, ReLU, SiLU, Sigmoid, Softmax, Softplus, Swish, Tanh};

// Convolutional layers
pub use conv::{Conv2d, Conv2dConfig, ConvTranspose2d};

// Pooling layers
pub use pooling::{AdaptiveAvgPool2d, AvgPool2d, GlobalAvgPool2d, MaxPool2d};

// Other layers
pub use attention::{
    MultiHeadAttention, MultiHeadAttentionConfig, AttentionMask,
    create_causal_mask, create_padding_mask_from_ids,
};
pub use batchnorm::BatchNorm;
pub use dropout::{Dropout, SpatialDropout};
pub use embedding::Embedding;
pub use feedforward::FeedForward;
pub use linear::Linear;
pub use norm::LayerNorm;
pub use positional::{
    LearnedPositionalEmbedding, SinusoidalPositionalEncoding, create_position_ids,
    RotaryPositionEmbedding, ALiBi,
};
pub use transformer::TransformerBlock;

// Base trait
pub use module::Module;