// --- Файл: src/serialization/mod.rs ---

//! Модуль для сериализации и десериализации моделей.
//!
//! Поддерживает несколько форматов:
//! - **SafeTensors**: Безопасный бинарный формат для хранения тензоров
//! - **JSON**: Человекочитаемый формат для метаданных и конфигурации
//! - **Checkpoint**: Полные снимки состояния модели и оптимизатора
//!
//! # Примеры
//!
//! ```rust,ignore
//! use rustyasg::serialization::{save_safetensors, load_safetensors};
//! use std::collections::HashMap;
//!
//! // Сохранение весов
//! let mut weights = HashMap::new();
//! weights.insert("layer1.weight".to_string(), tensor_data);
//! save_safetensors("model.safetensors", &weights)?;
//!
//! // Загрузка весов
//! let loaded = load_safetensors("model.safetensors")?;
//! ```

pub mod checkpoint;
pub mod safetensors_io;

pub use checkpoint::{Checkpoint, CheckpointConfig, save_checkpoint, load_checkpoint};
pub use safetensors_io::{save_safetensors, load_safetensors, SafeTensorsError};
