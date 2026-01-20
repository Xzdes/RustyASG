// --- Файл: src/serialization/safetensors_io.rs ---

//! Модуль для работы с форматом SafeTensors.
//!
//! SafeTensors - это безопасный и эффективный формат для хранения тензоров,
//! разработанный HuggingFace. Он обеспечивает:
//! - Быструю загрузку через memory-mapping
//! - Защиту от arbitrary code execution
//! - Поддержку различных типов данных

use crate::asg::Value;
use ndarray::ArrayD;
use safetensors::tensor::{SafeTensors, TensorView};
use safetensors::serialize_to_file;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;
use thiserror::Error;

/// Ошибки при работе с SafeTensors
#[derive(Error, Debug)]
pub enum SafeTensorsError {
    #[error("Ошибка ввода/вывода: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Ошибка SafeTensors: {0}")]
    SafeTensorsError(#[from] safetensors::SafeTensorError),

    #[error("Неподдерживаемый тип данных: {0}")]
    UnsupportedDtype(String),

    #[error("Ошибка формы тензора: ожидалось {expected:?}, получено {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Тензор '{0}' не найден")]
    TensorNotFound(String),
}

type Result<T> = std::result::Result<T, SafeTensorsError>;

/// Сохраняет HashMap с Value::Tensor в файл SafeTensors.
///
/// # Аргументы
///
/// * `path` - Путь к файлу для сохранения
/// * `tensors` - HashMap с именами тензоров и их значениями
///
/// # Пример
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
pub fn save_safetensors<P: AsRef<Path>>(
    path: P,
    tensors: &HashMap<String, Value>,
) -> Result<()> {
    let mut tensor_views: Vec<(&str, TensorView<'_>)> = Vec::new();
    let mut data_storage: HashMap<String, Vec<u8>> = HashMap::new();

    // Сначала конвертируем все данные в байты и сохраняем
    for (name, value) in tensors {
        if let Value::Tensor(arr) = value {
            let data: Vec<u8> = arr
                .iter()
                .flat_map(|&x| x.to_le_bytes())
                .collect();
            data_storage.insert(name.clone(), data);
        }
    }

    // Теперь создаем TensorView ссылающиеся на data_storage
    for (name, value) in tensors {
        if let Value::Tensor(arr) = value {
            let shape: Vec<usize> = arr.shape().to_vec();
            let data = data_storage.get(name).unwrap();
            tensor_views.push((
                name.as_str(),
                TensorView::new(
                    safetensors::Dtype::F32,
                    shape,
                    data,
                )?,
            ));
        }
    }

    // Сериализуем в файл
    serialize_to_file(tensor_views, &None, path.as_ref())?;

    Ok(())
}

/// Загружает тензоры из файла SafeTensors.
///
/// # Аргументы
///
/// * `path` - Путь к файлу SafeTensors
///
/// # Возвращает
///
/// HashMap с именами тензоров и их значениями
///
/// # Пример
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
    // Читаем файл в память
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Парсим SafeTensors
    let tensors = SafeTensors::deserialize(&buffer)?;

    let mut result = HashMap::new();

    for (name, tensor) in tensors.tensors() {
        // Проверяем тип данных
        match tensor.dtype() {
            safetensors::Dtype::F32 => {
                let shape: Vec<usize> = tensor.shape().to_vec();
                let data = tensor.data();

                // Конвертируем байты обратно в f32
                let floats: Vec<f32> = data
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();

                let floats_len = floats.len();

                // Создаем ArrayD
                let arr = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), floats)
                    .map_err(|_| SafeTensorsError::ShapeMismatch {
                        expected: shape.clone(),
                        actual: vec![floats_len],
                    })?;

                result.insert(name.to_string(), Value::Tensor(arr));
            }
            safetensors::Dtype::F64 => {
                let shape: Vec<usize> = tensor.shape().to_vec();
                let data = tensor.data();

                // Конвертируем f64 в f32
                let floats: Vec<f32> = data
                    .chunks_exact(8)
                    .map(|chunk| {
                        let val = f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7],
                        ]);
                        val as f32
                    })
                    .collect();

                let floats_len = floats.len();

                let arr = ArrayD::from_shape_vec(ndarray::IxDyn(&shape), floats)
                    .map_err(|_| SafeTensorsError::ShapeMismatch {
                        expected: shape.clone(),
                        actual: vec![floats_len],
                    })?;

                result.insert(name.to_string(), Value::Tensor(arr));
            }
            safetensors::Dtype::F16 | safetensors::Dtype::BF16 => {
                // Для F16/BF16 нужна дополнительная обработка
                // Пока просто выдаем ошибку
                return Err(SafeTensorsError::UnsupportedDtype(
                    format!("{:?}", tensor.dtype()),
                ));
            }
            other => {
                return Err(SafeTensorsError::UnsupportedDtype(format!("{:?}", other)));
            }
        }
    }

    Ok(result)
}

/// Загружает конкретный тензор по имени из файла SafeTensors.
pub fn load_tensor<P: AsRef<Path>>(path: P, name: &str) -> Result<Value> {
    let tensors = load_safetensors(path)?;
    tensors
        .get(name)
        .cloned()
        .ok_or_else(|| SafeTensorsError::TensorNotFound(name.to_string()))
}

/// Проверяет, содержит ли файл SafeTensors указанные тензоры.
pub fn contains_tensors<P: AsRef<Path>>(path: P, names: &[&str]) -> Result<bool> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let tensors = SafeTensors::deserialize(&buffer)?;
    let tensor_names: Vec<String> = tensors.names().iter().map(|s| s.to_string()).collect();

    Ok(names.iter().all(|name| tensor_names.contains(&name.to_string())))
}

/// Возвращает список имен тензоров в файле SafeTensors.
pub fn list_tensors<P: AsRef<Path>>(path: P) -> Result<Vec<String>> {
    let mut file = File::open(path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    let tensors = SafeTensors::deserialize(&buffer)?;
    Ok(tensors.names().iter().map(|s| s.to_string()).collect())
}

/// Возвращает информацию о тензорах в файле (имя, форма, тип).
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
            Value::Tensor(ArrayD::from_shape_vec(
                ndarray::IxDyn(&[2, 3]),
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ).unwrap()),
        );
        weights.insert(
            "test.bias".to_string(),
            Value::Tensor(ArrayD::from_shape_vec(
                ndarray::IxDyn(&[3]),
                vec![0.1, 0.2, 0.3],
            ).unwrap()),
        );

        let path = "test_safetensors.safetensors";

        // Сохраняем
        save_safetensors(path, &weights).expect("Failed to save");

        // Загружаем
        let loaded = load_safetensors(path).expect("Failed to load");

        // Проверяем
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

        // Удаляем тестовый файл
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

        let path = "test_list.safetensors";
        save_safetensors(path, &weights).expect("Failed to save");

        let names = list_tensors(path).expect("Failed to list");
        assert_eq!(names.len(), 2);
        assert!(names.contains(&"layer1.weight".to_string()));
        assert!(names.contains(&"layer2.weight".to_string()));

        fs::remove_file(path).ok();
    }
}
