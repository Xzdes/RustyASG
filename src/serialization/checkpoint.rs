//! Module for creating and loading model checkpoints.
//!
//! A checkpoint includes:
//! - Model weights (in SafeTensors format)
//! - Optimizer state
//! - Training metadata (epoch, loss, etc.)
//! - Model configuration

use super::safetensors_io::{load_safetensors, save_safetensors, SafeTensorsError};
use crate::asg::Value;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use thiserror::Error;

/// Errors that may occur while working with checkpoints.
#[derive(Error, Debug)]
pub enum CheckpointError {
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("SafeTensors error: {0}")]
    SafeTensorsError(#[from] SafeTensorsError),

    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    #[error("Checkpoint directory does not exist: {0}")]
    DirectoryNotFound(PathBuf),

    #[error("File not found: {0}")]
    FileNotFound(PathBuf),

    #[error("Invalid checkpoint format")]
    InvalidFormat,
}

type Result<T> = std::result::Result<T, CheckpointError>;

/// Checkpoint configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointConfig {
    /// Checkpoint format version.
    pub version: String,
    /// Model name.
    pub model_name: Option<String>,
    /// Current epoch.
    pub epoch: usize,
    /// Global training step.
    pub global_step: usize,
    /// Current learning rate.
    pub learning_rate: f32,
    /// Most recent loss value.
    pub last_loss: Option<f32>,
    /// Best observed loss value.
    pub best_loss: Option<f32>,
    /// Additional metadata.
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
    /// Creates a new checkpoint configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the model name.
    pub fn with_model_name(mut self, name: &str) -> Self {
        self.model_name = Some(name.to_string());
        self
    }

    /// Sets the epoch.
    pub fn with_epoch(mut self, epoch: usize) -> Self {
        self.epoch = epoch;
        self
    }

    /// Sets the global step.
    pub fn with_global_step(mut self, step: usize) -> Self {
        self.global_step = step;
        self
    }

    /// Sets the learning rate.
    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Sets the last loss value.
    pub fn with_last_loss(mut self, loss: f32) -> Self {
        self.last_loss = Some(loss);
        self
    }

    /// Sets the best loss value.
    pub fn with_best_loss(mut self, loss: f32) -> Self {
        self.best_loss = Some(loss);
        self
    }

    /// Adds a metadata entry.
    pub fn with_metadata(mut self, key: &str, value: &str) -> Self {
        self.metadata.insert(key.to_string(), value.to_string());
        self
    }
}

/// Optimizer state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    /// Optimizer type (SGD, Adam, etc.).
    pub optimizer_type: String,
    /// Optimizer parameters.
    pub params: HashMap<String, f64>,
    /// Per-parameter state (moments, etc.).
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

/// A complete model checkpoint.
#[derive(Debug)]
pub struct Checkpoint {
    /// Checkpoint configuration.
    pub config: CheckpointConfig,
    /// Model weights.
    pub model_weights: HashMap<String, Value>,
    /// Optional optimizer state.
    pub optimizer_state: Option<OptimizerState>,
}

impl Checkpoint {
    /// Creates a new checkpoint.
    pub fn new(weights: HashMap<String, Value>, config: CheckpointConfig) -> Self {
        Self {
            config,
            model_weights: weights,
            optimizer_state: None,
        }
    }

    /// Attaches optimizer state to the checkpoint.
    pub fn with_optimizer_state(mut self, state: OptimizerState) -> Self {
        self.optimizer_state = Some(state);
        self
    }
}

/// Saves a checkpoint to a directory.
///
/// Directory layout:
/// ```text
/// checkpoint_dir/
/// ├── config.json          # Configuration and metadata
/// ├── model.safetensors    # Model weights
/// └── optimizer.json       # Optimizer state (optional)
/// ```
pub fn save_checkpoint<P: AsRef<Path>>(path: P, checkpoint: &Checkpoint) -> Result<()> {
    let dir = path.as_ref();

    // Create the directory if it does not exist.
    fs::create_dir_all(dir)?;

    // Save the configuration.
    let config_path = dir.join("config.json");
    let config_json = serde_json::to_string_pretty(&checkpoint.config)?;
    let mut config_file = File::create(&config_path)?;
    config_file.write_all(config_json.as_bytes())?;

    // Save the model weights.
    let weights_path = dir.join("model.safetensors");
    save_safetensors(&weights_path, &checkpoint.model_weights)?;

    // Save the optimizer state if present.
    if let Some(ref opt_state) = checkpoint.optimizer_state {
        let opt_path = dir.join("optimizer.json");
        let opt_json = serde_json::to_string_pretty(opt_state)?;
        let mut opt_file = File::create(&opt_path)?;
        opt_file.write_all(opt_json.as_bytes())?;
    }

    Ok(())
}

/// Loads a checkpoint from a directory.
pub fn load_checkpoint<P: AsRef<Path>>(path: P) -> Result<Checkpoint> {
    let dir = path.as_ref();

    if !dir.exists() {
        return Err(CheckpointError::DirectoryNotFound(dir.to_path_buf()));
    }

    // Load the configuration.
    let config_path = dir.join("config.json");
    if !config_path.exists() {
        return Err(CheckpointError::FileNotFound(config_path));
    }
    let mut config_file = File::open(&config_path)?;
    let mut config_str = String::new();
    config_file.read_to_string(&mut config_str)?;
    let config: CheckpointConfig = serde_json::from_str(&config_str)?;

    // Load the weights.
    let weights_path = dir.join("model.safetensors");
    if !weights_path.exists() {
        return Err(CheckpointError::FileNotFound(weights_path));
    }
    let model_weights = load_safetensors(&weights_path)?;

    // Load the optimizer state if present.
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

/// Checkpoint manager for automatic saving.
pub struct CheckpointManager {
    /// Base directory for checkpoints.
    pub base_dir: PathBuf,
    /// Maximum number of checkpoints to keep.
    pub max_to_keep: usize,
    /// List of existing checkpoints.
    checkpoints: Vec<PathBuf>,
}

impl CheckpointManager {
    /// Creates a new checkpoint manager.
    pub fn new<P: AsRef<Path>>(base_dir: P, max_to_keep: usize) -> Self {
        Self {
            base_dir: base_dir.as_ref().to_path_buf(),
            max_to_keep,
            checkpoints: Vec::new(),
        }
    }

    /// Saves a checkpoint with an auto-generated name.
    pub fn save(&mut self, checkpoint: &Checkpoint) -> Result<PathBuf> {
        let checkpoint_name = format!(
            "checkpoint_epoch{}_step{}",
            checkpoint.config.epoch, checkpoint.config.global_step
        );
        let checkpoint_path = self.base_dir.join(&checkpoint_name);

        save_checkpoint(&checkpoint_path, checkpoint)?;
        self.checkpoints.push(checkpoint_path.clone());

        // Remove old checkpoints when the retention limit is exceeded.
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

    /// Loads the latest checkpoint.
    pub fn load_latest(&self) -> Result<Option<Checkpoint>> {
        if self.checkpoints.is_empty() {
            // Try to locate checkpoints in the base directory.
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

    /// Scans the base directory for existing checkpoints.
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

    /// Returns the path to the best checkpoint (the one with the minimum loss).
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

/// Fast save of model weights only.
pub fn save_weights<P: AsRef<Path>>(path: P, weights: &HashMap<String, Value>) -> Result<()> {
    save_safetensors(path, weights)?;
    Ok(())
}

/// Fast load of model weights only.
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
        assert_eq!(
            config.metadata.get("test_key"),
            Some(&"test_value".to_string())
        );
    }

    #[test]
    fn test_save_load_checkpoint() {
        let mut weights = HashMap::new();
        weights.insert(
            "layer.weight".to_string(),
            Value::Tensor(
                ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 2]), vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            ),
        );

        let config = CheckpointConfig::new()
            .with_model_name("test")
            .with_epoch(5);

        let checkpoint = Checkpoint::new(weights.clone(), config);

        // Use a unique path under the OS temp directory so parallel test runs
        // (`cargo test` runs in parallel by default) don't race on Windows.
        let path = std::env::temp_dir().join(format!(
            "rustyasg_ckpt_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        save_checkpoint(&path, &checkpoint).expect("Failed to save checkpoint");
        let loaded = load_checkpoint(&path).expect("Failed to load checkpoint");

        assert_eq!(loaded.config.model_name, Some("test".to_string()));
        assert_eq!(loaded.config.epoch, 5);
        assert!(loaded.model_weights.contains_key("layer.weight"));

        fs::remove_dir_all(&path).ok();
    }

    #[test]
    fn test_optimizer_state() {
        let mut state = OptimizerState::new("Adam");
        state.params.insert("lr".to_string(), 0.001);
        state.params.insert("beta1".to_string(), 0.9);
        state
            .state
            .insert("layer.weight_m".to_string(), vec![0.0; 4]);

        assert_eq!(state.optimizer_type, "Adam");
        assert_eq!(state.params.get("lr"), Some(&0.001));
    }
}
