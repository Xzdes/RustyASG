// --- Файл: src/serialization/checkpoint.rs ---

//! Модуль для создания и загрузки чекпоинтов модели.
//!
//! Чекпоинт включает:
//! - Веса модели (в формате SafeTensors)
//! - Состояние оптимизатора
//! - Метаданные обучения (эпоха, loss, и т.д.)
//! - Конфигурацию модели

use crate::asg::Value;
use super::safetensors_io::{save_safetensors, load_safetensors, SafeTensorsError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Ошибки при работе с чекпоинтами
#[derive(Error, Debug)]
pub enum CheckpointError {
    #[error("Ошибка ввода/вывода: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Ошибка SafeTensors: {0}")]
    SafeTensorsError(#[from] SafeTensorsError),

    #[error("Ошибка JSON: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Директория чекпоинта не существует: {0}")]
    DirectoryNotFound(PathBuf),

    #[error("Файл не найден: {0}")]
    FileNotFound(PathBuf),

    #[error("Неверный формат чекпоинта")]
    InvalidFormat,
}

type Result<T> = std::result::Result<T, CheckpointError>;

/// Конфигурация чекпоинта
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Версия формата чекпоинта
    pub version: String,
    /// Название модели
    pub model_name: Option<String>,
    /// Текущая эпоха
    pub epoch: usize,
    /// Глобальный шаг обучения
    pub global_step: usize,
    /// Текущий learning rate
    pub learning_rate: f32,
    /// Последнее значение loss
    pub last_loss: Option<f32>,
    /// Лучшее значение loss
    pub best_loss: Option<f32>,
    /// Дополнительные метаданные
    pub metadata: HashMap<String, String>,
}

impl Default for CheckpointConfig {
    fn default() -> Self {
        Self {
            version: "1.0".to_string(),
            model_name: None,
            epoch: 0,
            global_step: 0,
            learning_rate: 0.001,
            last_loss: None,
            best_loss: None,
            metadata: HashMap::new(),
        }
    }
}

impl CheckpointConfig {
    /// Создает новую конфигурацию чекпоинта
    pub fn new() -> Self {
        Self::default()
    }

    /// Устанавливает название модели
    pub fn with_model_name(mut self, name: &str) -> Self {
        self.model_name = Some(name.to_string());
        self
    }

    /// Устанавливает эпоху
    pub fn with_epoch(mut self, epoch: usize) -> Self {
        self.epoch = epoch;
        self
    }

    /// Устанавливает глобальный шаг
    pub fn with_global_step(mut self, step: usize) -> Self {
        self.global_step = step;
        self
    }

    /// Устанавливает learning rate
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Устанавливает последний loss
    pub fn with_last_loss(mut self, loss: f32) -> Self {
        self.last_loss = Some(loss);
        self
    }

    /// Устанавливает лучший loss
    pub fn with_best_loss(mut self, loss: f32) -> Self {
        self.best_loss = Some(loss);
        self
    }

    /// Добавляет метаданные
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Состояние оптимизатора
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    /// Тип оптимизатора (SGD, Adam, и т.д.)
    pub optimizer_type: String,
    /// Параметры оптимизатора
    pub params: HashMap<String, f64>,
    /// Состояние для каждого параметра (моменты и т.д.)
    pub state: HashMap<String, Vec<f64>>,
}

impl OptimizerState {
    pub fn new(optimizer_type: &str) -> Self {
        Self {
            optimizer_type: optimizer_type.to_string(),
            params: HashMap::new(),
            state: HashMap::new(),
        }
    }
}

/// Полный чекпоинт модели
#[derive(Debug)]
pub struct Checkpoint {
    /// Конфигурация чекпоинта
    pub config: CheckpointConfig,
    /// Веса модели
    pub model_weights: HashMap<String, Value>,
    /// Состояние оптимизатора (опционально)
    pub optimizer_state: Option<OptimizerState>,
}

impl Checkpoint {
    /// Создает новый чекпоинт
    pub fn new(weights: HashMap<String, Value>, config: CheckpointConfig) -> Self {
        Self {
            config,
            model_weights: weights,
            optimizer_state: None,
        }
    }

    /// Добавляет состояние оптимизатора
    pub fn with_optimizer_state(mut self, state: OptimizerState) -> Self {
        self.optimizer_state = Some(state);
        self
    }
}

/// Сохраняет чекпоинт в директорию.
///
/// Структура директории:
/// ```text
/// checkpoint_dir/
/// ├── config.json          # Конфигурация и метаданные
/// ├── model.safetensors    # Веса модели
/// └── optimizer.json       # Состояние оптимизатора (опционально)
/// ```
pub fn save_checkpoint<P: AsRef<Path>>(path: P, checkpoint: &Checkpoint) -> Result<()> {
    let dir = path.as_ref();

    // Создаем директорию если не существует
    fs::create_dir_all(dir)?;

    // Сохраняем конфигурацию
    let config_path = dir.join("config.json");
    let config_json = serde_json::to_string_pretty(&checkpoint.config)?;
    let mut config_file = File::create(&config_path)?;
    config_file.write_all(config_json.as_bytes())?;

    // Сохраняем веса модели
    let weights_path = dir.join("model.safetensors");
    save_safetensors(&weights_path, &checkpoint.model_weights)?;

    // Сохраняем состояние оптимизатора если есть
    if let Some(ref opt_state) = checkpoint.optimizer_state {
        let opt_path = dir.join("optimizer.json");
        let opt_json = serde_json::to_string_pretty(opt_state)?;
        let mut opt_file = File::create(&opt_path)?;
        opt_file.write_all(opt_json.as_bytes())?;
    }

    Ok(())
}

/// Загружает чекпоинт из директории.
pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> Result<Checkpoint> {
    let dir = path.as_ref();

    if !dir.exists() {
        return Err(CheckpointError::DirectoryNotFound(dir.to_path_buf()));
    }

    // Загружаем конфигурацию
    let config_path = dir.join("config.json");
    if !config_path.exists() {
        return Err(CheckpointError::FileNotFound(config_path));
    }
    let mut config_file = File::open(&config_path)?;
    let mut config_str = String::new();
    config_file.read_to_string(&mut config_str)?;
    let config: CheckpointConfig = serde_json::from_str(&config_str)?;

    // Загружаем веса
    let weights_path = dir.join("model.safetensors");
    if !weights_path.exists() {
        return Err(CheckpointError::FileNotFound(weights_path));
    }
    let model_weights = load_safetensors(&weights_path)?;

    // Загружаем состояние оптимизатора если есть
    let opt_path = dir.join("optimizer.json");
    let optimizer_state = if opt_path.exists() {
        let mut opt_file = File::open(&opt_path)?;
        let mut opt_str = String::new();
        opt_file.read_to_string(&mut opt_str)?;
        Some(serde_json::from_str(&opt_str)?)
    } else {
        None
    };

    Ok(Checkpoint {
        config,
        model_weights,
        optimizer_state,
    })
}

/// Менеджер чекпоинтов для автоматического сохранения.
pub struct CheckpointManager {
    /// Базовая директория для чекпоинтов
    pub base_dir: PathBuf,
    /// Максимальное количество сохраняемых чекпоинтов
    pub max_to_keep: usize,
    /// Список существующих чекпоинтов
    checkpoints: Vec<PathBuf>,
}

impl CheckpointManager {
    /// Создает новый менеджер чекпоинтов.
    pub fn new<P: AsRef<Path>>(base_dir: P, max_to_keep: usize) -> Self {
        Self {
            base_dir: base_dir.as_ref().to_path_buf(),
            max_to_keep,
            checkpoints: Vec::new(),
        }
    }

    /// Сохраняет чекпоинт с автоматическим именем.
    pub fn save(&mut self, checkpoint: &Checkpoint) -> Result<PathBuf> {
        let checkpoint_name = format!(
            "checkpoint_epoch{}_step{}",
            checkpoint.config.epoch, checkpoint.config.global_step
        );
        let checkpoint_path = self.base_dir.join(&checkpoint_name);

        save_checkpoint(&checkpoint_path, checkpoint)?;
        self.checkpoints.push(checkpoint_path.clone());

        // Удаляем старые чекпоинты если превышен лимит
        while self.checkpoints.len() > self.max_to_keep {
            if let Some(old_path) = self.checkpoints.first() {
                if old_path.exists() {
                    fs::remove_dir_all(old_path)?;
                }
            }
            self.checkpoints.remove(0);
        }

        Ok(checkpoint_path)
    }

    /// Загружает последний чекпоинт.
    pub fn load_latest(&self) -> Result<Option<Checkpoint>> {
        if self.checkpoints.is_empty() {
            // Попробуем найти чекпоинты в директории
            let checkpoints = self.find_checkpoints()?;
            if checkpoints.is_empty() {
                return Ok(None);
            }
            let latest = checkpoints.last().unwrap();
            return Ok(Some(load_checkpoint(latest)?));
        }

        let latest = self.checkpoints.last().unwrap();
        Ok(Some(load_checkpoint(latest)?))
    }

    /// Ищет существующие чекпоинты в базовой директории.
    fn find_checkpoints(&self) -> Result<Vec<PathBuf>> {
        if !self.base_dir.exists() {
            return Ok(Vec::new());
        }

        let mut checkpoints: Vec<PathBuf> = fs::read_dir(&self.base_dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.is_dir()
                    && path
                        .file_name()
                        .map(|n| n.to_string_lossy().starts_with("checkpoint_"))
                        .unwrap_or(false)
            })
            .collect();

        checkpoints.sort();
        Ok(checkpoints)
    }

    /// Возвращает путь к лучшему чекпоинту (с минимальным loss).
    pub fn get_best_checkpoint(&self) -> Result<Option<PathBuf>> {
        let checkpoints = self.find_checkpoints()?;
        let mut best_loss = f32::MAX;
        let mut best_path = None;

        for path in checkpoints {
            let checkpoint = load_checkpoint(&path)?;
            if let Some(loss) = checkpoint.config.best_loss {
                if loss < best_loss {
                    best_loss = loss;
                    best_path = Some(path);
                }
            }
        }

        Ok(best_path)
    }
}

/// Быстрое сохранение только весов модели.
pub fn save_weights<P: AsRef<Path>>(path: P, weights: &HashMap<String, Value>) -> Result<()> {
    save_safetensors(path, weights)?;
    Ok(())
}

/// Быстрая загрузка только весов модели.
pub fn load_weights<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Value>> {
    let weights = load_safetensors(path)?;
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::ArrayD;

    #[test]
    fn test_checkpoint_config() {
        let config = CheckpointConfig::new()
            .with_model_name("test_model")
            .with_epoch(10)
            .with_global_step(1000)
            .with_learning_rate(0.001)
            .with_last_loss(0.5)
            .with_metadata("test_key", "test_value");

        assert_eq!(config.model_name, Some("test_model".to_string()));
        assert_eq!(config.epoch, 10);
        assert_eq!(config.global_step, 1000);
        assert!((config.learning_rate - 0.001).abs() < 1e-6);
        assert_eq!(config.last_loss, Some(0.5));
        assert_eq!(config.metadata.get("test_key"), Some(&"test_value".to_string()));
    }

    #[test]
    fn test_save_load_checkpoint() {
        let mut weights = HashMap::new();
        weights.insert(
            "layer.weight".to_string(),
            Value::Tensor(ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 2]),
                vec![1.0, 2.0, 3.0, 4.0],
            ).unwrap()),
        );

        let config = CheckpointConfig::new()
            .with_model_name("test")
            .with_epoch(5);

        let checkpoint = Checkpoint::new(weights.clone(), config);

        let path = "test_checkpoint_dir";

        // Сохраняем
        save_checkpoint(path, &checkpoint).expect("Failed to save checkpoint");

        // Загружаем
        let loaded = load_checkpoint(path).expect("Failed to load checkpoint");

        assert_eq!(loaded.config.model_name, Some("test".to_string()));
        assert_eq!(loaded.config.epoch, 5);
        assert!(loaded.model_weights.contains_key("layer.weight"));

        // Очистка
        fs::remove_dir_all(path).ok();
    }

    #[test]
    fn test_optimizer_state() {
        let mut state = OptimizerState::new("Adam");
        state.params.insert("lr".to_string(), 0.001);
        state.params.insert("beta1".to_string(), 0.9);
        state.state.insert("layer.weight_m".to_string(), vec![0.0; 4]);

        assert_eq!(state.optimizer_type, "Adam");
        assert_eq!(state.params.get("lr"), Some(&0.001));
    }
}
