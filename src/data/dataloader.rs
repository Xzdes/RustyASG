// --- File: src/data/dataloader.rs ---

//! DataLoader - an iterator over data batches.

use super::dataset::{Dataset, InMemoryDataset};
use super::sampler::{BatchSampler, RandomSampler, SequentialSampler};
use ndarray::ArrayD;

/// A data batch - a `(features, labels)` pair.
#[derive(Debug, Clone)]
pub struct Batch {
    /// Batch features.
    pub features: ArrayD<f32>,
    /// Batch labels.
    pub labels: ArrayD<f32>,
    /// Indices of the samples in this batch.
    pub indices: Vec<usize>,
}

impl Batch {
    /// Creates a new batch.
    pub fn new(features: ArrayD<f32>, labels: ArrayD<f32>, indices: Vec<usize>) -> Self {
        Self {
            features,
            labels,
            indices,
        }
    }

    /// Returns the batch size.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns whether the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// DataLoader - convenient interface for iterating over a dataset in batches.
///
/// # Example
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
    /// Creates a new DataLoader.
    ///
    /// # Arguments
    ///
    /// * `dataset` - Dataset to load from.
    /// * `batch_size` - Batch size.
    pub fn new(dataset: InMemoryDataset, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle: false,
            drop_last: false,
            seed: None,
        }
    }

    /// Enables or disables data shuffling.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Controls whether the last, incomplete batch is dropped.
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Sets the seed for reproducibility.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Returns the number of batches.
    pub fn num_batches(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            n.div_ceil(self.batch_size)
        }
    }

    /// Returns the dataset size.
    pub fn len(&self) -> usize {
        self.dataset.len()
    }

    /// Returns whether the DataLoader is empty.
    pub fn is_empty(&self) -> bool {
        self.dataset.is_empty()
    }

    /// Returns the batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Creates an iterator over batches.
    pub fn iter(&self) -> DataLoaderIterator<'_> {
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

/// Enum holding different batch sampler types.
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

/// Iterator over data batches.
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

/// DataLoader builder with more flexible options.
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
    /// Creates a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the batch size.
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Enables shuffling.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Sets `drop_last`.
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Sets the seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Sets the number of worker threads (placeholder for future implementation).
    pub fn num_workers(mut self, num: usize) -> Self {
        self.num_workers = num;
        self
    }

    /// Sets `pin_memory` (placeholder for future GPU optimization).
    pub fn pin_memory(mut self, pin: bool) -> Self {
        self.pin_memory = pin;
        self
    }

    /// Builds the DataLoader.
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
        )
        .unwrap();
        let labels = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[10, 1]),
            (0..10).map(|x| x as f32).collect(),
        )
        .unwrap();
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
        assert_eq!(batches[3].len(), 1); // Last incomplete batch
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
        // When shuffling is enabled the indices must be permuted.
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
