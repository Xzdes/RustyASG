//! Module containing building blocks for neural networks.
//!
//! In the graph-based architecture, each "layer" is a constructor
//! that adds a specific pattern of nodes (operations and parameters)
//! to the Abstract Semantic Graph (ASG).

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