// --- Файл: src/data/mod.rs ---

//! Модуль для работы с данными: Dataset и DataLoader API.
//!
//! Предоставляет абстракции для загрузки и батчинга данных аналогично PyTorch.
//!
//! # Основные компоненты
//!
//! - **Dataset**: Трейт для определения источников данных
//! - **DataLoader**: Итератор по батчам данных с поддержкой shuffle
//! - **Sampler**: Стратегии сэмплирования данных
//! - **Transform**: Преобразования данных
//!
//! # Пример использования
//!
//! ```rust,ignore
//! use rustyasg::data::{Dataset, DataLoader, InMemoryDataset};
//!
//! // Создаем датасет
//! let dataset = InMemoryDataset::new(features, labels);
//!
//! // Создаем DataLoader
//! let loader = DataLoader::new(dataset, 32)
//!     .shuffle(true)
//!     .drop_last(false);
//!
//! // Итерируем по батчам
//! for (batch_x, batch_y) in loader.iter() {
//!     // Обучение на батче
//! }
//! ```

pub mod dataset;
pub mod dataloader;
pub mod sampler;
pub mod transforms;

pub use dataset::{Dataset, InMemoryDataset, MapDataset};
pub use dataloader::{DataLoader, Batch};
pub use sampler::{Sampler, SequentialSampler, RandomSampler, BatchSampler};
pub use transforms::{Transform, Compose, Normalize, RandomNoise};
