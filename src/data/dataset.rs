// --- Файл: src/data/dataset.rs ---

//! Определение трейта Dataset и базовые реализации.

use ndarray::ArrayD;
use std::sync::Arc;

/// Трейт для источников данных.
///
/// Каждый датасет должен уметь:
/// - Возвращать количество элементов
/// - Возвращать элемент по индексу
pub trait Dataset: Send + Sync {
    /// Тип элемента данных (features)
    type Item;
    /// Тип метки (label)
    type Label;

    /// Возвращает количество элементов в датасете.
    fn len(&self) -> usize;

    /// Проверяет, пуст ли датасет.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Возвращает элемент и метку по индексу.
    fn get(&self, index: usize) -> Option<(Self::Item, Self::Label)>;

    /// Возвращает только элемент данных по индексу.
    fn get_item(&self, index: usize) -> Option<Self::Item> {
        self.get(index).map(|(item, _)| item)
    }

    /// Возвращает только метку по индексу.
    fn get_label(&self, index: usize) -> Option<Self::Label> {
        self.get(index).map(|(_, label)| label)
    }
}

/// Датасет, хранящий данные в памяти.
///
/// Простейшая реализация датасета для работы с данными,
/// которые полностью помещаются в оперативную память.
#[derive(Debug, Clone)]
pub struct InMemoryDataset {
    /// Признаки (features) - каждая строка это один образец
    features: Arc<ArrayD<f32>>,
    /// Метки (labels) - каждая строка соответствует образцу
    labels: Arc<ArrayD<f32>>,
    /// Количество образцов
    num_samples: usize,
}

impl InMemoryDataset {
    /// Создает новый датасет из массивов признаков и меток.
    ///
    /// # Аргументы
    ///
    /// * `features` - Массив признаков формы [num_samples, ...]
    /// * `labels` - Массив меток формы [num_samples, ...]
    ///
    /// # Паника
    ///
    /// Паникует если количество образцов в features и labels не совпадает.
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

    /// Создает датасет только с признаками (для unsupervised learning).
    pub fn from_features(features: ArrayD<f32>) -> Self {
        let num_samples = features.shape()[0];
        let labels = ArrayD::zeros(ndarray::IxDyn(&[num_samples]));
        Self {
            features: Arc::new(features),
            labels: Arc::new(labels),
            num_samples,
        }
    }

    /// Возвращает форму признаков (без batch dimension).
    pub fn feature_shape(&self) -> Vec<usize> {
        self.features.shape()[1..].to_vec()
    }

    /// Возвращает форму меток (без batch dimension).
    pub fn label_shape(&self) -> Vec<usize> {
        self.labels.shape()[1..].to_vec()
    }

    /// Возвращает срез признаков по индексам.
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

    /// Возвращает срез меток по индексам.
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

/// Датасет с применением функции преобразования.
///
/// Позволяет применять трансформации к данным на лету.
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
    /// Создает новый MapDataset с преобразованиями.
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
        self.inner.get(index).map(|(item, label)| {
            ((self.item_transform)(item), (self.label_transform)(label))
        })
    }
}

/// Датасет, объединяющий несколько датасетов.
pub struct ConcatDataset<D: Dataset> {
    datasets: Vec<D>,
    cumulative_sizes: Vec<usize>,
    total_len: usize,
}

impl<D: Dataset> ConcatDataset<D> {
    /// Создает объединенный датасет из списка датасетов.
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

    /// Находит индекс датасета и локальный индекс для глобального индекса.
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

/// Датасет с подмножеством индексов.
pub struct SubsetDataset<D: Dataset> {
    inner: D,
    indices: Vec<usize>,
}

impl<D: Dataset> SubsetDataset<D> {
    /// Создает подмножество датасета по указанным индексам.
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

/// Разделяет датасет на train/validation/test части.
pub fn train_test_split<D: Dataset>(
    dataset: D,
    train_ratio: f32,
    shuffle: bool,
    seed: Option<u64>,
) -> (SubsetDataset<D>, SubsetDataset<D>)
where
    D: Clone,
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
            let mut rng = rand::thread_rng();
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
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        ).unwrap();
        let labels = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4, 1]),
            vec![0.0, 1.0, 0.0, 1.0],
        ).unwrap();

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
        ).unwrap();
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
