// --- Файл: src/data/dataloader.rs ---

//! DataLoader - итератор по батчам данных.

use super::dataset::{Dataset, InMemoryDataset};
use super::sampler::{BatchSampler, RandomSampler, Sampler, SequentialSampler};
use ndarray::ArrayD;

/// Батч данных - пара (features, labels).
#[derive(Debug, Clone)]
pub struct Batch {
    /// Признаки батча
    pub features: ArrayD<f32>,
    /// Метки батча
    pub labels: ArrayD<f32>,
    /// Индексы образцов в этом батче
    pub indices: Vec<usize>,
}

impl Batch {
    /// Создает новый батч.
    pub fn new(features: ArrayD<f32>, labels: ArrayD<f32>, indices: Vec<usize>) -> Self {
        Self {
            features,
            labels,
            indices,
        }
    }

    /// Возвращает размер батча.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Проверяет, пуст ли батч.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// DataLoader - удобный интерфейс для итерации по датасету батчами.
///
/// # Пример
///
/// ```rust,ignore
/// let loader = DataLoader::new(dataset, 32)
///     .shuffle(true)
///     .drop_last(false);
///
/// for batch in loader.iter() {
///     println!("Batch size: {}", batch.len());
/// }
/// ```
pub struct DataLoader {
    dataset: InMemoryDataset,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    seed: Option<u64>,
}

impl DataLoader {
    /// Создает новый DataLoader.
    ///
    /// # Аргументы
    ///
    /// * `dataset` - Датасет для загрузки
    /// * `batch_size` - Размер батча
    pub fn new(dataset: InMemoryDataset, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle: false,
            drop_last: false,
            seed: None,
        }
    }

    /// Включает/выключает перемешивание данных.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Устанавливает, нужно ли отбрасывать последний неполный батч.
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Устанавливает seed для воспроизводимости.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Возвращает количество батчей.
    pub fn num_batches(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            (n + self.batch_size - 1) / self.batch_size
        }
    }

    /// Возвращает размер датасета.
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Проверяет, пуст ли DataLoader.
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }

    /// Возвращает размер батча.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Создает итератор по батчам.
    pub fn iter(&self) -> DataLoaderIterator {
        if self.shuffle {
            let sampler = if let Some(seed) = self.seed {
                RandomSampler::with_seed(self.dataset.len(), seed)
            } else {
                RandomSampler::new(self.dataset.len())
            };
            let batch_sampler = BatchSampler::new(sampler, self.batch_size, self.drop_last);
            DataLoaderIterator {
                dataset: &self.dataset,
                batch_sampler: BatchSamplerEnum::Random(batch_sampler),
            }
        } else {
            let sampler = SequentialSampler::new(self.dataset.len());
            let batch_sampler = BatchSampler::new(sampler, self.batch_size, self.drop_last);
            DataLoaderIterator {
                dataset: &self.dataset,
                batch_sampler: BatchSamplerEnum::Sequential(batch_sampler),
            }
        }
    }
}

/// Enum для хранения разных типов batch sampler'ов.
enum BatchSamplerEnum {
    Sequential(BatchSampler<SequentialSampler>),
    Random(BatchSampler<RandomSampler>),
}

impl Iterator for BatchSamplerEnum {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            BatchSamplerEnum::Sequential(s) => s.next(),
            BatchSamplerEnum::Random(s) => s.next(),
        }
    }
}

/// Итератор по батчам данных.
pub struct DataLoaderIterator<'a> {
    dataset: &'a InMemoryDataset,
    batch_sampler: BatchSamplerEnum,
}

impl<'a> Iterator for DataLoaderIterator<'a> {
    type Item = Batch;

    fn next(&mut self) -> Option<Self::Item> {
        let indices = self.batch_sampler.next()?;

        let features = self.dataset.get_features_batch(&indices);
        let labels = self.dataset.get_labels_batch(&indices);

        Some(Batch::new(features, labels, indices))
    }
}

/// Конструктор DataLoader с более гибкими опциями.
pub struct DataLoaderBuilder {
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    seed: Option<u64>,
    num_workers: usize,
    pin_memory: bool,
}

impl Default for DataLoaderBuilder {
    fn default() -> Self {
        Self {
            batch_size: 1,
            shuffle: false,
            drop_last: false,
            seed: None,
            num_workers: 0,
            pin_memory: false,
        }
    }
}

impl DataLoaderBuilder {
    /// Создает новый builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Устанавливает размер батча.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Включает перемешивание.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Устанавливает drop_last.
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Устанавливает seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Устанавливает количество worker потоков (placeholder для будущей реализации).
    pub fn num_workers(mut self, num: usize) -> Self {
        self.num_workers = num;
        self
    }

    /// Устанавливает pin_memory (placeholder для будущей GPU оптимизации).
    pub fn pin_memory(mut self, pin: bool) -> Self {
        self.pin_memory = pin;
        self
    }

    /// Строит DataLoader.
    pub fn build(self, dataset: InMemoryDataset) -> DataLoader {
        let mut loader = DataLoader::new(dataset, self.batch_size)
            .shuffle(self.shuffle)
            .drop_last(self.drop_last);

        if let Some(seed) = self.seed {
            loader = loader.seed(seed);
        }

        loader
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_dataset() -> InMemoryDataset {
        let features = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[10, 4]),
            (0..40).map(|x| x as f32).collect(),
        ).unwrap();
        let labels = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[10, 1]),
            (0..10).map(|x| x as f32).collect(),
        ).unwrap();
        InMemoryDataset::new(features, labels)
    }

    #[test]
    fn test_dataloader_basic() {
        let dataset = create_test_dataset();
        let loader = DataLoader::new(dataset, 3);

        assert_eq!(loader.num_batches(), 4);
        assert_eq!(loader.len(), 10);

        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 4);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[3].len(), 1); // Последний неполный батч
    }

    #[test]
    fn test_dataloader_drop_last() {
        let dataset = create_test_dataset();
        let loader = DataLoader::new(dataset, 3).drop_last(true);

        let batches: Vec<_> = loader.iter().collect();
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn test_dataloader_shuffle() {
        let dataset = create_test_dataset();
        let loader = DataLoader::new(dataset, 10).shuffle(true).seed(42);

        let batch = loader.iter().next().unwrap();
        // При shuffle индексы должны быть перемешаны
        assert_eq!(batch.len(), 10);
    }

    #[test]
    fn test_dataloader_builder() {
        let dataset = create_test_dataset();
        let loader = DataLoaderBuilder::new()
            .batch_size(4)
            .shuffle(true)
            .drop_last(false)
            .seed(123)
            .build(dataset);

        assert_eq!(loader.batch_size(), 4);
        assert_eq!(loader.num_batches(), 3);
    }

    #[test]
    fn test_batch_features_shape() {
        let dataset = create_test_dataset();
        let loader = DataLoader::new(dataset, 3);

        let batch = loader.iter().next().unwrap();
        assert_eq!(batch.features.shape(), &[3, 4]);
        assert_eq!(batch.labels.shape(), &[3, 1]);
    }
}
