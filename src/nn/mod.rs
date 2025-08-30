//! Модуль, содержащий строительные блоки для нейронных сетей.
//!
//! В графовой архитектуре, каждый "слой" является конструктором,
//! который добавляет определенный паттерн узлов (операций и параметров)
//! в Абстрактный Семантический Граф (ASG).

// Объявляем все субмодули, которые мы создали.
pub mod activations;
pub mod attention;
pub mod feedforward;
pub mod linear;
pub mod module;
pub mod norm;
pub mod transformer;

// Ре-экспортируем самые важные структуры для удобства использования.
// Убираем неиспользуемый Sigmoid, чтобы избавиться от предупреждений.
pub use activations::ReLU;
pub use attention::MultiHeadAttention;
pub use feedforward::FeedForward;
pub use linear::Linear;
pub use module::Module;
pub use norm::LayerNorm;
pub use transformer::TransformerBlock;