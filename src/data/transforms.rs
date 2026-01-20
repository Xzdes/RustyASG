// --- Файл: src/data/transforms.rs ---

//! Преобразования данных для предобработки.

use ndarray::ArrayD;
use rand::Rng;
use rand::SeedableRng;

/// Трейт для преобразований данных.
pub trait Transform: Send + Sync {
    /// Применяет преобразование к данным.
    fn apply(&self, data: ArrayD<f32>) -> ArrayD<f32>;

    /// Применяет преобразование на месте (если возможно).
    fn apply_inplace(&self, data: &mut ArrayD<f32>) {
        *data = self.apply(data.clone());
    }
}

/// Композиция нескольких преобразований.
pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    /// Создает пустую композицию.
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Создает композицию из списка преобразований.
    pub fn from_transforms(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }

    /// Добавляет преобразование в композицию.
    pub fn add<T: Transform + 'static>(mut self, transform: T) -> Self {
        self.transforms.push(Box::new(transform));
        self
    }
}

impl Default for Compose {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for Compose {
    fn apply(&self, mut data: ArrayD<f32>) -> ArrayD<f32> {
        for transform in &self.transforms {
            data = transform.apply(data);
        }
        data
    }
}

/// Нормализация данных: (x - mean) / std
pub struct Normalize {
    mean: ArrayD<f32>,
    std: ArrayD<f32>,
}

impl Normalize {
    /// Создает нормализатор с заданными параметрами.
    pub fn new(mean: ArrayD<f32>, std: ArrayD<f32>) -> Self {
        Self { mean, std }
    }

    /// Создает нормализатор для одномерных данных.
    pub fn from_scalars(mean: f32, std: f32) -> Self {
        Self {
            mean: ArrayD::from_elem(ndarray::IxDyn(&[]), mean),
            std: ArrayD::from_elem(ndarray::IxDyn(&[]), std),
        }
    }

    /// Создает нормализатор для данных изображений (3 канала).
    pub fn imagenet() -> Self {
        Self {
            mean: ArrayD::from_shape_vec(
                ndarray::IxDyn(&[3]),
                vec![0.485, 0.456, 0.406],
            ).unwrap(),
            std: ArrayD::from_shape_vec(
                ndarray::IxDyn(&[3]),
                vec![0.229, 0.224, 0.225],
            ).unwrap(),
        }
    }

    /// Вычисляет параметры нормализации из данных.
    pub fn fit(data: &ArrayD<f32>) -> Self {
        let mean_val = data.mean().unwrap_or(0.0);
        let std_val = data.std(0.0);

        Self {
            mean: ArrayD::from_elem(ndarray::IxDyn(&[]), mean_val),
            std: ArrayD::from_elem(ndarray::IxDyn(&[]), std_val.max(1e-8)),
        }
    }
}

impl Transform for Normalize {
    fn apply(&self, data: ArrayD<f32>) -> ArrayD<f32> {
        // Простая нормализация для скалярных mean/std
        if self.mean.len() == 1 && self.std.len() == 1 {
            let mean = self.mean.first().unwrap();
            let std = self.std.first().unwrap();
            return data.mapv(|x| (x - mean) / std);
        }

        // Для векторных mean/std нужно broadcasting
        // Это упрощенная версия
        data.mapv(|x| {
            let mean = self.mean.first().unwrap_or(&0.0);
            let std = self.std.first().unwrap_or(&1.0);
            (x - mean) / std
        })
    }
}

/// Добавление случайного шума.
pub struct RandomNoise {
    std: f32,
    seed: Option<u64>,
}

impl RandomNoise {
    /// Создает преобразование с гауссовым шумом.
    pub fn new(std: f32) -> Self {
        Self { std, seed: None }
    }

    /// Устанавливает seed для воспроизводимости.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl Transform for RandomNoise {
    fn apply(&self, mut data: ArrayD<f32>) -> ArrayD<f32> {
        use rand::Rng;

        if let Some(s) = self.seed {
            let mut rng = rand::rngs::StdRng::seed_from_u64(s);
            for x in data.iter_mut() {
                // Box-Muller transform для гауссова шума
                let u1: f32 = rng.random();
                let u2: f32 = rng.random();
                let z = (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).cos();
                *x += z * self.std;
            }
        } else {
            let mut rng = rand::thread_rng();
            for x in data.iter_mut() {
                let u1: f32 = rng.random();
                let u2: f32 = rng.random();
                let z = (-2.0_f32 * u1.ln()).sqrt() * (2.0_f32 * std::f32::consts::PI * u2).cos();
                *x += z * self.std;
            }
        }

        data
    }
}

/// Масштабирование данных в диапазон [min, max].
pub struct MinMaxScale {
    min_val: f32,
    max_val: f32,
}

impl MinMaxScale {
    /// Создает масштабирование в диапазон [0, 1].
    pub fn new() -> Self {
        Self {
            min_val: 0.0,
            max_val: 1.0,
        }
    }

    /// Создает масштабирование в заданный диапазон.
    pub fn with_range(min_val: f32, max_val: f32) -> Self {
        Self { min_val, max_val }
    }
}

impl Default for MinMaxScale {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for MinMaxScale {
    fn apply(&self, data: ArrayD<f32>) -> ArrayD<f32> {
        let data_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
        let data_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let range = data_max - data_min;
        if range.abs() < 1e-8 {
            return data;
        }

        let target_range = self.max_val - self.min_val;
        data.mapv(|x| {
            let normalized = (x - data_min) / range;
            normalized * target_range + self.min_val
        })
    }
}

/// Стандартизация (z-score): (x - mean) / std из данных.
pub struct StandardScale;

impl StandardScale {
    pub fn new() -> Self {
        Self
    }
}

impl Default for StandardScale {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for StandardScale {
    fn apply(&self, data: ArrayD<f32>) -> ArrayD<f32> {
        let mean = data.mean().unwrap_or(0.0);
        let std = data.std(0.0).max(1e-8);
        data.mapv(|x| (x - mean) / std)
    }
}

/// Clipping значений в диапазон.
pub struct Clip {
    min_val: f32,
    max_val: f32,
}

impl Clip {
    /// Создает clipping преобразование.
    pub fn new(min_val: f32, max_val: f32) -> Self {
        Self { min_val, max_val }
    }
}

impl Transform for Clip {
    fn apply(&self, data: ArrayD<f32>) -> ArrayD<f32> {
        data.mapv(|x| x.clamp(self.min_val, self.max_val))
    }
}

/// Логарифмическое преобразование: log(x + eps).
pub struct Log {
    eps: f32,
}

impl Log {
    /// Создает log преобразование.
    pub fn new() -> Self {
        Self { eps: 1e-8 }
    }

    /// Устанавливает eps для численной стабильности.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
}

impl Default for Log {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for Log {
    fn apply(&self, data: ArrayD<f32>) -> ArrayD<f32> {
        data.mapv(|x| (x + self.eps).ln())
    }
}

/// Flatten - преобразует многомерный тензор в одномерный.
pub struct Flatten;

impl Flatten {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for Flatten {
    fn apply(&self, data: ArrayD<f32>) -> ArrayD<f32> {
        let total: usize = data.len();
        let (vec, _offset) = data.into_raw_vec_and_offset();
        ArrayD::from_shape_vec(ndarray::IxDyn(&[total]), vec).unwrap()
    }
}

/// One-Hot Encoding для меток классов.
pub struct OneHot {
    num_classes: usize,
}

impl OneHot {
    /// Создает one-hot encoder.
    pub fn new(num_classes: usize) -> Self {
        Self { num_classes }
    }
}

impl Transform for OneHot {
    fn apply(&self, data: ArrayD<f32>) -> ArrayD<f32> {
        let len = data.len();
        let mut result = ArrayD::zeros(ndarray::IxDyn(&[len, self.num_classes]));

        for (i, &class_idx) in data.iter().enumerate() {
            let idx = class_idx as usize;
            if idx < self.num_classes {
                result[[i, idx]] = 1.0;
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize() {
        let data = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![1.0, 2.0, 3.0, 4.0],
        ).unwrap();

        let norm = Normalize::from_scalars(2.5, 1.0);
        let result = norm.apply(data);

        assert!((result[0] - (-1.5)).abs() < 1e-6);
        assert!((result[3] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_compose() {
        let data = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![0.0, 10.0, 20.0, 30.0],
        ).unwrap();

        let transform = Compose::new()
            .add(MinMaxScale::new())
            .add(StandardScale::new());

        let result = transform.apply(data);
        assert_eq!(result.len(), 4);
    }

    #[test]
    fn test_min_max_scale() {
        let data = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![0.0, 25.0, 50.0, 100.0],
        ).unwrap();

        let scale = MinMaxScale::new();
        let result = scale.apply(data);

        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[3] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_clip() {
        let data = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[5]),
            vec![-10.0, 0.0, 5.0, 10.0, 20.0],
        ).unwrap();

        let clip = Clip::new(0.0, 10.0);
        let result = clip.apply(data);

        assert_eq!(result[0], 0.0);
        assert_eq!(result[2], 5.0);
        assert_eq!(result[4], 10.0);
    }

    #[test]
    fn test_one_hot() {
        let data = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[3]),
            vec![0.0, 2.0, 1.0],
        ).unwrap();

        let one_hot = OneHot::new(3);
        let result = one_hot.apply(data);

        assert_eq!(result.shape(), &[3, 3]);
        assert_eq!(result[[0, 0]], 1.0);
        assert_eq!(result[[1, 2]], 1.0);
        assert_eq!(result[[2, 1]], 1.0);
    }

    #[test]
    fn test_flatten() {
        let data = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[2, 3]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        ).unwrap();

        let flatten = Flatten::new();
        let result = flatten.apply(data);

        assert_eq!(result.shape(), &[6]);
    }
}
