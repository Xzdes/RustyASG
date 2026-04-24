//! Module for working with the SafeTensors format.
//!
//! SafeTensors is a safe and efficient tensor storage format developed by
//! HuggingFace. It provides:
//! - Fast loading via memory-mapping
//! - Protection from arbitrary code execution
//! - Support for multiple data types

use crate::asg::Value;
use ndarray::ArrayD;
use safetensors::serialize_to_file;
use safetensors::tensor::{SafeTensors, TensorView};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use thiserror::Error;

/// Errors that may occur while working with SafeTensors.
#[derive(Error, Debug)]
pub enum SafeTensorsError {
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("SafeTensors error: {0}")]
    SafeTensorsError(#[from] safetensors::SafeTensorError),

    #[error("Unsupported data type: {0}")]
    UnsupportedDtype(String),

    #[error("Tensor shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Tensor '{0}' not found")]
    TensorNotFound(String),
}

type Result<T> = std::result::Result<T, SafeTensorsError>;

/// Saves a HashMap of `Value::Tensor` entries to a SafeTensors file.
///
/// # Arguments
///
/// * `path` - Destination file path.
/// * `tensors` - HashMap of tensor names and their values.
///
/// # Example
///
/// ```rust,ignore
/// use std::collections::HashMap;
/// use ndarray::ArrayD;
/// use rustyasg::asg::Value;
/// use rustyasg::serialization::save_safetensors;
///
/// let mut weights = HashMap::new();
/// weights.insert(
///     "layer.weight".to_string(),
///     Value::Tensor(ArrayD::zeros(ndarray::IxDyn(&[4, 4]))),
/// );
/// save_safetensors("weights.safetensors", &weights)?;
/// ```
pub fn save_safetensors<P: AsRef<Path>>(path: P, tensors: &HashMap<String, Value>) -> Result<()> {
    let mut tensor_views: Vec<(&str, TensorView<'_>)> = Vec::new();
    let mut data_storage: HashMap<String, Vec<u8>> = HashMap::new();

    // First convert all data to bytes and store it.
    for (name, value) in tensors {
        if let Value::Tensor(arr) = value {
            let data: Vec<u8> = arr.iter().flat_map(|&x| x.to_le_bytes()).collect();
            data_storage.insert(name.clone(), data);
        }
    }

    // Then create TensorView values that reference data_storage.
    for (name, value) in tensors {
        if let Value::Tensor(arr) = value {
            let shape: Vec<usize> = arr.shape().to_vec();
            let data = data_storage.get(name).unwrap();
            tensor_views.push((
                name.as_str(),
                TensorView::new(safetensors::Dtype::F32, shape, data)?,
            ));
        }
    }

    // Serialize to file.
    serialize_to_file(tensor_views, &None, path.as_ref())?;

    Ok(())
}

/// Loads tensors from a SafeTensors file.
///
/// # Arguments
///
/// * `path` - Path to the SafeTensors file.
///
/// # Returns
///
/// A HashMap of tensor names and their values.
///
/// # Example
///
/// ```rust,ignore
/// use rustyasg::serialization::load_safetensors;
///
/// let weights = load_safetensors("weights.safetensors")?;
/// for (name, value) in &weights {
///     println!("Loaded tensor: {}", name);
/// }
/// ```
pub fn load_safetensors<P: AsRef<Path>>(path: P) -> Result<HashMap<String, Value>> {
    // Read the file into memory.
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Parse SafeTensors.
    let tensors = SafeTensors::deserialize(&buffer)?;

    let mut result = HashMap::new();

    for (name, tensor) in tensors.tensors() {
        // Check the data type.
        match tensor.dtype() {
            safetensors::Dtype::F32 => {
                let shape: Vec<usize> = tensor.shape().to_vec();
                let data = tensor.data();

                // Convert bytes back to f32.
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                let floats_len = floats.len();

                // Create the ArrayD.
                let arr = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), floats).map_err(|_| {
                    SafeTensorsError::ShapeMismatch {
                        expected: shape.clone(),
                        actual: vec![floats_len],
                    }
                })?;

                result.insert(name.to_string(), Value::Tensor(arr));
            }
            safetensors::Dtype::F64 => {
                let shape: Vec<usize> = tensor.shape().to_vec();
                let data = tensor.data();

                // Convert f64 to f32.
                let floats: Vec<f32> = data
                    .chunks_exact(8)
                    .map(|chunk| {
                        let val = f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ]);
                        val as f32
                    })
                    .collect();

                let floats_len = floats.len();

                let arr = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), floats).map_err(|_| {
                    SafeTensorsError::ShapeMismatch {
                        expected: shape.clone(),
                        actual: vec![floats_len],
                    }
                })?;

                result.insert(name.to_string(), Value::Tensor(arr));
            }
            safetensors::Dtype::F16 | safetensors::Dtype::BF16 => {
                // F16/BF16 requires additional handling.
                // For now we just return an error.
                return Err(SafeTensorsError::UnsupportedDtype(format!(
                    "{:?}",
                    tensor.dtype()
                )));
            }
            other => {
                return Err(SafeTensorsError::UnsupportedDtype(format!("{:?}", other)));
            }
        }
    }

    Ok(result)
}

/// Loads a specific tensor by name from a SafeTensors file.
pub fn load_tensor<P: AsRef<Path>>(path: P, name: &str) -> Result<Value> {
    let tensors = load_safetensors(path)?;
    tensors
        .get(name)
        .cloned()
        .ok_or_else(|| SafeTensorsError::TensorNotFound(name.to_string()))
}

/// Checks whether a SafeTensors file contains the specified tensors.
pub fn contains_tensors<P: AsRef<Path>>(path: P, names: &[&str]) -> Result<bool> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let tensors = SafeTensors::deserialize(&buffer)?;
    let tensor_names: Vec<String> = tensors.names().iter().map(|s| s.to_string()).collect();

    Ok(names
        .iter()
        .all(|name| tensor_names.contains(&name.to_string())))
}

/// Returns the list of tensor names in a SafeTensors file.
pub fn list_tensors<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let tensors = SafeTensors::deserialize(&buffer)?;
    Ok(tensors.names().iter().map(|s| s.to_string()).collect())
}

/// Returns information about the tensors in the file (name, shape, dtype).
pub fn tensor_info<P: AsRef<Path>>(path: P) -> Result<Vec<(String, Vec<usize>, String)>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let tensors = SafeTensors::deserialize(&buffer)?;
    let mut info = Vec::new();

    for (name, tensor) in tensors.tensors() {
        info.push((
            name.to_string(),
            tensor.shape().to_vec(),
            format!("{:?}", tensor.dtype()),
        ));
    }

    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_save_load_safetensors() {
        let mut weights = HashMap::new();
        weights.insert(
            "test.weight".to_string(),
            Value::Tensor(
                ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
                    .unwrap(),
            ),
        );
        weights.insert(
            "test.bias".to_string(),
            Value::Tensor(
                ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.1, 0.2, 0.3]).unwrap(),
            ),
        );

        // Unique temp path to avoid races when tests run in parallel on Windows.
        let path = std::env::temp_dir().join(format!(
            "rustyasg_safetensors_{}_{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        save_safetensors(&path, &weights).expect("Failed to save");
        let loaded = load_safetensors(&path).expect("Failed to load");

        // Verify.
        assert_eq!(loaded.len(), 2);
        assert!(loaded.contains_key("test.weight"));
        assert!(loaded.contains_key("test.bias"));

        if let (Value::Tensor(original), Value::Tensor(loaded_tensor)) =
            (&weights["test.weight"], &loaded["test.weight"])
        {
            assert_eq!(original.shape(), loaded_tensor.shape());
            for (a, b) in original.iter().zip(loaded_tensor.iter()) {
                assert!((a - b).abs() < 1e-6);
            }
        }

        // Remove the test file.
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_list_tensors() {
        let mut weights = HashMap::new();
        weights.insert(
            "layer1.weight".to_string(),
            Value::Tensor(ArrayD::zeros(ndarray::IxDyn(&[4, 4]))),
        );
        weights.insert(
            "layer2.weight".to_string(),
            Value::Tensor(ArrayD::zeros(ndarray::IxDyn(&[4, 4]))),
        );

        let path = std::env::temp_dir().join(format!(
            "rustyasg_list_{}_{}.safetensors",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        save_safetensors(&path, &weights).expect("Failed to save");

        let names = list_tensors(&path).expect("Failed to list");
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"layer1.weight".to_string()));
        assert!(names.contains(&"layer2.weight".to_string()));

        fs::remove_file(&path).ok();
    }
}
