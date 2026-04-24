// --- File: src/data/dataset.rs ---

//! The `Dataset` trait and baseline implementations.

use ndarray::ArrayD;
use std::sync::Arc;

/// Trait for data sources.
///
/// Every dataset must be able to:
/// - Return the number of elements
/// - Return an element by index
pub trait Dataset: Send + Sync {
    /// Feature item type.
    type Item;
    /// Label type.
    type Label;

    /// Returns the number of elements in the dataset.
    fn len(&self) -> usize;

    /// Returns whether the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the item and label at the given index.
    fn get(&self, index: usize) -> Option<(Self::Item, Self::Label)>;

    /// Returns only the feature item at the given index.
    fn get_item(&self, index: usize) -> Option<Self::Item> {
        self.get(index).map(|(item, _)| item)
    }

    /// Returns only the label at the given index.
    fn get_label(&self, index: usize) -> Option<Self::Label> {
        self.get(index).map(|(_, label)| label)
    }
}

/// Dataset that stores all data in memory.
///
/// The simplest dataset implementation, suitable for data that fits
/// entirely in RAM.
#[derive(Debug, Clone)]
pub struct InMemoryDataset {
    /// Features - each row is one sample.
    features: Arc<ArrayD<f32>>,
    /// Labels - each row corresponds to a sample.
    labels: Arc<ArrayD<f32>>,
    /// Number of samples.
    num_samples: usize,
}

impl InMemoryDataset {
    /// Creates a new dataset from feature and label arrays.
    ///
    /// # Arguments
    ///
    /// * `features` - Feature array of shape `[num_samples, ...]`.
    /// * `labels` - Label array of shape `[num_samples, ...]`.
    ///
    /// # Panics
    ///
    /// Panics when the number of samples in `features` and `labels` does not match.
    pub fn new(features: ArrayD<f32>, labels: ArrayD<f32>) -> Self {
        let num_samples = features.shape()[0];
        assert_eq!(
            num_samples,
            labels.shape()[0],
            "Number of samples in features and labels must match"
        );

        Self {
            features: Arc::new(features),
            labels: Arc::new(labels),
            num_samples,
        }
    }

    /// Creates a dataset containing only features (for unsupervised learning).
    pub fn from_features(features: ArrayD<f32>) -> Self {
        let num_samples = features.shape()[0];
        let labels = ArrayD::zeros(ndarray::IxDyn(&[num_samples]));
        Self {
            features: Arc::new(features),
            labels: Arc::new(labels),
            num_samples,
        }
    }

    /// Returns the feature shape (without the batch dimension).
    pub fn feature_shape(&self) -> Vec<usize> {
        self.features.shape()[1..].to_vec()
    }

    /// Returns the label shape (without the batch dimension).
    pub fn label_shape(&self) -> Vec<usize> {
        self.labels.shape()[1..].to_vec()
    }

    /// Returns a slice of features for the given indices.
    pub fn get_features_batch(&self, indices: &[usize]) -> ArrayD<f32> {
        let feature_shape = self.feature_shape();
        let mut batch_shape = vec![indices.len()];
        batch_shape.extend(&feature_shape);

        let mut batch = ArrayD::zeros(ndarray::IxDyn(&batch_shape));

        for (i, &idx) in indices.iter().enumerate() {
            let sample = self.features.index_axis(ndarray::Axis(0), idx);
            batch.index_axis_mut(ndarray::Axis(0), i).assign(&sample);
        }

        batch
    }

    /// Returns a slice of labels for the given indices.
    pub fn get_labels_batch(&self, indices: &[usize]) -> ArrayD<f32> {
        let label_shape = self.label_shape();
        let mut batch_shape = vec![indices.len()];
        batch_shape.extend(&label_shape);

        let mut batch = ArrayD::zeros(ndarray::IxDyn(&batch_shape));

        for (i, &idx) in indices.iter().enumerate() {
            let sample = self.labels.index_axis(ndarray::Axis(0), idx);
            batch.index_axis_mut(ndarray::Axis(0), i).assign(&sample);
        }

        batch
    }
}

impl Dataset for InMemoryDataset {
    type Item = ArrayD<f32>;
    type Label = ArrayD<f32>;

    fn len(&self) -> usize {
        self.num_samples
    }

    fn get(&self, index: usize) -> Option<(Self::Item, Self::Label)> {
        if index >= self.num_samples {
            return None;
        }

        let feature = self.features.index_axis(ndarray::Axis(0), index).to_owned();
        let label = self.labels.index_axis(ndarray::Axis(0), index).to_owned();

        Some((feature, label))
    }
}

/// Dataset that applies transformation functions.
///
/// Allows transformations to be applied to data on the fly.
pub struct MapDataset<D, F, G>
where
    D: Dataset,
    F: Fn(D::Item) -> D::Item + Send + Sync,
    G: Fn(D::Label) -> D::Label + Send + Sync,
{
    inner: D,
    item_transform: F,
    label_transform: G,
}

impl<D, F, G> MapDataset<D, F, G>
where
    D: Dataset,
    F: Fn(D::Item) -> D::Item + Send + Sync,
    G: Fn(D::Label) -> D::Label + Send + Sync,
{
    /// Creates a new `MapDataset` with the given transformations.
    pub fn new(dataset: D, item_transform: F, label_transform: G) -> Self {
        Self {
            inner: dataset,
            item_transform,
            label_transform,
        }
    }
}

impl<D, F, G> Dataset for MapDataset<D, F, G>
where
    D: Dataset,
    F: Fn(D::Item) -> D::Item + Send + Sync,
    G: Fn(D::Label) -> D::Label + Send + Sync,
{
    type Item = D::Item;
    type Label = D::Label;

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn get(&self, index: usize) -> Option<(Self::Item, Self::Label)> {
        self.inner
            .get(index)
            .map(|(item, label)| ((self.item_transform)(item), (self.label_transform)(label)))
    }
}

/// Dataset that concatenates several datasets.
pub struct ConcatDataset<D: Dataset> {
    datasets: Vec<D>,
    cumulative_sizes: Vec<usize>,
    total_len: usize,
}

impl<D: Dataset> ConcatDataset<D> {
    /// Creates a concatenated dataset from a list of datasets.
    pub fn new(datasets: Vec<D>) -> Self {
        let mut cumulative_sizes = Vec::with_capacity(datasets.len());
        let mut total = 0;

        for dataset in &datasets {
            total += dataset.len();
            cumulative_sizes.push(total);
        }

        Self {
            datasets,
            cumulative_sizes,
            total_len: total,
        }
    }

    /// Finds the dataset index and local index for a given global index.
    fn find_dataset(&self, index: usize) -> Option<(usize, usize)> {
        if index >= self.total_len {
            return None;
        }

        let dataset_idx = self.cumulative_sizes.partition_point(|&x| x <= index);
        let local_idx = if dataset_idx == 0 {
            index
        } else {
            index - self.cumulative_sizes[dataset_idx - 1]
        };

        Some((dataset_idx, local_idx))
    }
}

impl<D: Dataset> Dataset for ConcatDataset<D> {
    type Item = D::Item;
    type Label = D::Label;

    fn len(&self) -> usize {
        self.total_len
    }

    fn get(&self, index: usize) -> Option<(Self::Item, Self::Label)> {
        let (dataset_idx, local_idx) = self.find_dataset(index)?;
        self.datasets[dataset_idx].get(local_idx)
    }
}

/// Dataset view over a subset of indices.
pub struct SubsetDataset<D: Dataset> {
    inner: D,
    indices: Vec<usize>,
}

impl<D: Dataset> SubsetDataset<D> {
    /// Creates a subset of the dataset over the specified indices.
    pub fn new(dataset: D, indices: Vec<usize>) -> Self {
        Self {
            inner: dataset,
            indices,
        }
    }
}

impl<D: Dataset> Dataset for SubsetDataset<D> {
    type Item = D::Item;
    type Label = D::Label;

    fn len(&self) -> usize {
        self.indices.len()
    }

    fn get(&self, index: usize) -> Option<(Self::Item, Self::Label)> {
        let actual_idx = *self.indices.get(index)?;
        self.inner.get(actual_idx)
    }
}

/// Splits a dataset into train/validation/test subsets.
pub fn train_test_split<D>(
    dataset: D,
    train_ratio: f32,
    shuffle: bool,
    seed: Option<u64>,
) -> (SubsetDataset<D>, SubsetDataset<D>)
where
    D: Dataset + Clone,
{
    let n = dataset.len();
    let train_size = (n as f32 * train_ratio) as usize;

    let mut indices: Vec<usize> = (0..n).collect();

    if shuffle {
        use rand::seq::SliceRandom;
        use rand::SeedableRng;

        if let Some(s) = seed {
            let mut rng = rand::rngs::StdRng::seed_from_u64(s);
            indices.shuffle(&mut rng);
        } else {
            let mut rng = rand::rng();
            indices.shuffle(&mut rng);
        }
    }

    let train_indices = indices[..train_size].to_vec();
    let test_indices = indices[train_size..].to_vec();

    (
        SubsetDataset::new(dataset.clone(), train_indices),
        SubsetDataset::new(dataset, test_indices),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_in_memory_dataset() {
        let features = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4, 3]),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let labels =
            ArrayD::from_shape_vec(ndarray::IxDyn(&[4, 1]), vec![0.0, 1.0, 0.0, 1.0]).unwrap();

        let dataset = InMemoryDataset::new(features, labels);

        assert_eq!(dataset.len(), 4);
        assert_eq!(dataset.feature_shape(), vec![3]);
        assert_eq!(dataset.label_shape(), vec![1]);

        let (item, label) = dataset.get(0).unwrap();
        assert_eq!(item.len(), 3);
        assert_eq!(label.len(), 1);
    }

    #[test]
    fn test_batch_retrieval() {
        let features = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4, 2]),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        let labels = ArrayD::zeros(ndarray::IxDyn(&[4]));

        let dataset = InMemoryDataset::new(features, labels);
        let batch = dataset.get_features_batch(&[0, 2]);

        assert_eq!(batch.shape(), &[2, 2]);
    }

    #[test]
    fn test_concat_dataset() {
        let features1 = ArrayD::zeros(ndarray::IxDyn(&[3, 2]));
        let labels1 = ArrayD::zeros(ndarray::IxDyn(&[3]));
        let dataset1 = InMemoryDataset::new(features1, labels1);

        let features2 = ArrayD::zeros(ndarray::IxDyn(&[2, 2]));
        let labels2 = ArrayD::zeros(ndarray::IxDyn(&[2]));
        let dataset2 = InMemoryDataset::new(features2, labels2);

        let concat = ConcatDataset::new(vec![dataset1, dataset2]);
        assert_eq!(concat.len(), 5);
    }
}
