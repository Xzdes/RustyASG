//! Модуль, содержащий строительные блоки для нейронных сетей.
//!
//! В графовой архитектуре, каждый "слой" является конструктором,
//! который добавляет определенный паттерн узлов (операций и параметров)
//! в Абстрактный Семантический Граф (ASG).

// Объявляем все субмодули
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

// Ре-экспортируем структуры для удобства использования

// Активации
pub use activations::{ELU, GELU, LeakyReLU, ReLU, SiLU, Sigmoid, Softmax, Softplus, Swish, Tanh};

// Сверточные слои
pub use conv::{Conv2d, Conv2dConfig, ConvTranspose2d};

// Pooling слои
pub use pooling::{AdaptiveAvgPool2d, AvgPool2d, GlobalAvgPool2d, MaxPool2d};

// Другие слои
pub use attention::MultiHeadAttention;
pub use batchnorm::BatchNorm;
pub use dropout::{Dropout, SpatialDropout};
pub use embedding::Embedding;
pub use feedforward::FeedForward;
pub use linear::Linear;
pub use norm::LayerNorm;
pub use positional::{LearnedPositionalEmbedding, SinusoidalPositionalEncoding, create_position_ids};
pub use transformer::TransformerBlock;

// Базовый трейт
pub use module::Module;