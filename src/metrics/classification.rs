// --- Файл: src/metrics/classification.rs ---

//! Метрики для задач классификации.

use super::Metric;
use ndarray::ArrayD;

/// Метрика точности (Accuracy) для классификации.
///
/// Accuracy = (TP + TN) / (TP + TN + FP + FN)
#[derive(Debug, Clone, Default)]
pub struct Accuracy {
    correct: usize,
    total: usize,
}

impl Accuracy {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Metric for Accuracy {
    type Prediction = ArrayD<f32>;
    type Target = ArrayD<f32>;
    type Output = f64;

    fn update(&mut self, predictions: &Self::Prediction, targets: &Self::Target) {
        // Для multi-class: берем argmax от predictions
        // Для binary: округляем predictions
        let is_multiclass = predictions.ndim() > 1 && predictions.shape().last().unwrap_or(&1) > &1;

        if is_multiclass {
            // Multi-class: [batch_size, num_classes]
            let batch_size = predictions.shape()[0];
            for i in 0..batch_size {
                let pred_class = (0..predictions.shape()[1])
                    .max_by(|&a, &b| {
                        predictions[[i, a]]
                            .partial_cmp(&predictions[[i, b]])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0);

                let target_class = if targets.ndim() > 1 && targets.shape().last().unwrap_or(&1) > &1 {
                    // One-hot encoded
                    (0..targets.shape()[1])
                        .max_by(|&a, &b| {
                            targets[[i, a]]
                                .partial_cmp(&targets[[i, b]])
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .unwrap_or(0)
                } else {
                    // Class indices
                    targets.as_slice().unwrap()[i] as usize
                };

                if pred_class == target_class {
                    self.correct += 1;
                }
                self.total += 1;
            }
        } else {
            // Binary: [batch_size] or [batch_size, 1]
            for (pred, target) in predictions.iter().zip(targets.iter()) {
                let pred_class = if *pred >= 0.5 { 1 } else { 0 };
                let target_class = if *target >= 0.5 { 1 } else { 0 };
                if pred_class == target_class {
                    self.correct += 1;
                }
                self.total += 1;
            }
        }
    }

    fn compute(&self) -> Self::Output {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }

    fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }

    fn name(&self) -> &str {
        "Accuracy"
    }
}

/// Confusion Matrix для бинарной классификации.
#[derive(Debug, Clone, Default)]
pub struct BinaryConfusionMatrix {
    pub true_positives: usize,
    pub true_negatives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
}

impl BinaryConfusionMatrix {
    pub fn new() -> Self {
        Self::default()
    }

    /// Обновляет матрицу ошибок.
    pub fn update(&mut self, predictions: &ArrayD<f32>, targets: &ArrayD<f32>, threshold: f32) {
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let pred_positive = *pred >= threshold;
            let actual_positive = *target >= 0.5;

            match (pred_positive, actual_positive) {
                (true, true) => self.true_positives += 1,
                (true, false) => self.false_positives += 1,
                (false, true) => self.false_negatives += 1,
                (false, false) => self.true_negatives += 1,
            }
        }
    }

    /// Сбрасывает состояние.
    pub fn reset(&mut self) {
        self.true_positives = 0;
        self.true_negatives = 0;
        self.false_positives = 0;
        self.false_negatives = 0;
    }

    /// Общее количество образцов.
    pub fn total(&self) -> usize {
        self.true_positives + self.true_negatives + self.false_positives + self.false_negatives
    }

    /// Вычисляет accuracy.
    pub fn accuracy(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }
        (self.true_positives + self.true_negatives) as f64 / total as f64
    }

    /// Вычисляет precision.
    pub fn precision(&self) -> f64 {
        let denom = self.true_positives + self.false_positives;
        if denom == 0 {
            return 0.0;
        }
        self.true_positives as f64 / denom as f64
    }

    /// Вычисляет recall (sensitivity).
    pub fn recall(&self) -> f64 {
        let denom = self.true_positives + self.false_negatives;
        if denom == 0 {
            return 0.0;
        }
        self.true_positives as f64 / denom as f64
    }

    /// Вычисляет specificity.
    pub fn specificity(&self) -> f64 {
        let denom = self.true_negatives + self.false_positives;
        if denom == 0 {
            return 0.0;
        }
        self.true_negatives as f64 / denom as f64
    }

    /// Вычисляет F1-Score.
    pub fn f1_score(&self) -> f64 {
        let p = self.precision();
        let r = self.recall();
        if p + r == 0.0 {
            return 0.0;
        }
        2.0 * p * r / (p + r)
    }
}

/// Метрика Precision для бинарной классификации.
#[derive(Debug, Clone)]
pub struct Precision {
    confusion: BinaryConfusionMatrix,
    threshold: f32,
}

impl Precision {
    pub fn new() -> Self {
        Self {
            confusion: BinaryConfusionMatrix::new(),
            threshold: 0.5,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }
}

impl Default for Precision {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for Precision {
    type Prediction = ArrayD<f32>;
    type Target = ArrayD<f32>;
    type Output = f64;

    fn update(&mut self, predictions: &Self::Prediction, targets: &Self::Target) {
        self.confusion.update(predictions, targets, self.threshold);
    }

    fn compute(&self) -> Self::Output {
        self.confusion.precision()
    }

    fn reset(&mut self) {
        self.confusion.reset();
    }

    fn name(&self) -> &str {
        "Precision"
    }
}

/// Метрика Recall для бинарной классификации.
#[derive(Debug, Clone)]
pub struct Recall {
    confusion: BinaryConfusionMatrix,
    threshold: f32,
}

impl Recall {
    pub fn new() -> Self {
        Self {
            confusion: BinaryConfusionMatrix::new(),
            threshold: 0.5,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }
}

impl Default for Recall {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for Recall {
    type Prediction = ArrayD<f32>;
    type Target = ArrayD<f32>;
    type Output = f64;

    fn update(&mut self, predictions: &Self::Prediction, targets: &Self::Target) {
        self.confusion.update(predictions, targets, self.threshold);
    }

    fn compute(&self) -> Self::Output {
        self.confusion.recall()
    }

    fn reset(&mut self) {
        self.confusion.reset();
    }

    fn name(&self) -> &str {
        "Recall"
    }
}

/// Метрика F1-Score для бинарной классификации.
#[derive(Debug, Clone)]
pub struct F1Score {
    confusion: BinaryConfusionMatrix,
    threshold: f32,
}

impl F1Score {
    pub fn new() -> Self {
        Self {
            confusion: BinaryConfusionMatrix::new(),
            threshold: 0.5,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }
}

impl Default for F1Score {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for F1Score {
    type Prediction = ArrayD<f32>;
    type Target = ArrayD<f32>;
    type Output = f64;

    fn update(&mut self, predictions: &Self::Prediction, targets: &Self::Target) {
        self.confusion.update(predictions, targets, self.threshold);
    }

    fn compute(&self) -> Self::Output {
        self.confusion.f1_score()
    }

    fn reset(&mut self) {
        self.confusion.reset();
    }

    fn name(&self) -> &str {
        "F1Score"
    }
}

/// Confusion Matrix для многоклассовой классификации.
#[derive(Debug, Clone)]
pub struct MultiClassConfusionMatrix {
    num_classes: usize,
    matrix: Vec<Vec<usize>>, // [actual][predicted]
}

impl MultiClassConfusionMatrix {
    pub fn new(num_classes: usize) -> Self {
        Self {
            num_classes,
            matrix: vec![vec![0; num_classes]; num_classes],
        }
    }

    /// Обновляет матрицу ошибок.
    pub fn update(&mut self, predictions: &ArrayD<f32>, targets: &ArrayD<f32>) {
        let batch_size = predictions.shape()[0];

        for i in 0..batch_size {
            // Находим предсказанный класс (argmax)
            let pred_class = (0..self.num_classes)
                .max_by(|&a, &b| {
                    predictions[[i, a]]
                        .partial_cmp(&predictions[[i, b]])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);

            // Находим истинный класс
            let actual_class = if targets.ndim() > 1 && targets.shape()[1] == self.num_classes {
                // One-hot encoded
                (0..self.num_classes)
                    .max_by(|&a, &b| {
                        targets[[i, a]]
                            .partial_cmp(&targets[[i, b]])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0)
            } else {
                // Class indices
                targets.as_slice().unwrap()[i] as usize
            };

            if actual_class < self.num_classes && pred_class < self.num_classes {
                self.matrix[actual_class][pred_class] += 1;
            }
        }
    }

    /// Сбрасывает состояние.
    pub fn reset(&mut self) {
        for row in &mut self.matrix {
            for cell in row {
                *cell = 0;
            }
        }
    }

    /// Возвращает матрицу ошибок.
    pub fn get_matrix(&self) -> &Vec<Vec<usize>> {
        &self.matrix
    }

    /// Вычисляет accuracy.
    pub fn accuracy(&self) -> f64 {
        let correct: usize = (0..self.num_classes).map(|i| self.matrix[i][i]).sum();
        let total: usize = self.matrix.iter().flat_map(|row| row.iter()).sum();
        if total == 0 {
            return 0.0;
        }
        correct as f64 / total as f64
    }

    /// Вычисляет precision для каждого класса.
    pub fn precision_per_class(&self) -> Vec<f64> {
        (0..self.num_classes)
            .map(|c| {
                let tp = self.matrix[c][c];
                let predicted_as_c: usize = (0..self.num_classes).map(|a| self.matrix[a][c]).sum();
                if predicted_as_c == 0 {
                    0.0
                } else {
                    tp as f64 / predicted_as_c as f64
                }
            })
            .collect()
    }

    /// Вычисляет recall для каждого класса.
    pub fn recall_per_class(&self) -> Vec<f64> {
        (0..self.num_classes)
            .map(|c| {
                let tp = self.matrix[c][c];
                let actual_c: usize = self.matrix[c].iter().sum();
                if actual_c == 0 {
                    0.0
                } else {
                    tp as f64 / actual_c as f64
                }
            })
            .collect()
    }

    /// Вычисляет macro-averaged F1-Score.
    pub fn macro_f1(&self) -> f64 {
        let precisions = self.precision_per_class();
        let recalls = self.recall_per_class();

        let f1_scores: Vec<f64> = precisions
            .iter()
            .zip(recalls.iter())
            .map(|(&p, &r)| {
                if p + r == 0.0 {
                    0.0
                } else {
                    2.0 * p * r / (p + r)
                }
            })
            .collect();

        f1_scores.iter().sum::<f64>() / self.num_classes as f64
    }
}

/// Top-K Accuracy для многоклассовой классификации.
#[derive(Debug, Clone)]
pub struct TopKAccuracy {
    k: usize,
    correct: usize,
    total: usize,
}

impl TopKAccuracy {
    pub fn new(k: usize) -> Self {
        Self {
            k,
            correct: 0,
            total: 0,
        }
    }
}

impl Metric for TopKAccuracy {
    type Prediction = ArrayD<f32>;
    type Target = ArrayD<f32>;
    type Output = f64;

    fn update(&mut self, predictions: &Self::Prediction, targets: &Self::Target) {
        let batch_size = predictions.shape()[0];
        let num_classes = predictions.shape()[1];

        for i in 0..batch_size {
            // Находим top-k классов
            let mut scores: Vec<(usize, f32)> = (0..num_classes)
                .map(|c| (c, predictions[[i, c]]))
                .collect();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top_k: Vec<usize> = scores.iter().take(self.k).map(|(c, _)| *c).collect();

            // Находим истинный класс
            let actual_class = if targets.ndim() > 1 && targets.shape()[1] == num_classes {
                (0..num_classes)
                    .max_by(|&a, &b| {
                        targets[[i, a]]
                            .partial_cmp(&targets[[i, b]])
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap_or(0)
            } else {
                targets.as_slice().unwrap()[i] as usize
            };

            if top_k.contains(&actual_class) {
                self.correct += 1;
            }
            self.total += 1;
        }
    }

    fn compute(&self) -> Self::Output {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }

    fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }

    fn name(&self) -> &str {
        "TopKAccuracy"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_binary() {
        let mut acc = Accuracy::new();

        let preds = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![0.9, 0.8, 0.3, 0.1],
        ).unwrap();
        let targets = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![1.0, 1.0, 0.0, 0.0],
        ).unwrap();

        acc.update(&preds, &targets);
        assert!((acc.compute() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_binary_confusion_matrix() {
        let mut cm = BinaryConfusionMatrix::new();

        let preds = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![0.9, 0.8, 0.3, 0.6],
        ).unwrap();
        let targets = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![1.0, 0.0, 0.0, 1.0],
        ).unwrap();

        cm.update(&preds, &targets, 0.5);

        assert_eq!(cm.true_positives, 2);  // 0.9->1, 0.6->1
        assert_eq!(cm.false_positives, 1); // 0.8->0 but pred 1
        assert_eq!(cm.true_negatives, 1);  // 0.3->0
        assert_eq!(cm.false_negatives, 0);
    }

    #[test]
    fn test_precision_recall() {
        let mut precision = Precision::new();
        let mut recall = Recall::new();

        let preds = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[6]),
            vec![0.9, 0.8, 0.7, 0.3, 0.2, 0.6],
        ).unwrap();
        let targets = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[6]),
            vec![1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        ).unwrap();

        precision.update(&preds, &targets);
        recall.update(&preds, &targets);

        // TP=2 (0.9, 0.8), FP=2 (0.7, 0.6), FN=1 (0.2)
        // Precision = 2/4 = 0.5
        // Recall = 2/3 ≈ 0.667
        assert!((precision.compute() - 0.5).abs() < 1e-6);
        assert!((recall.compute() - 2.0/3.0).abs() < 1e-6);
    }

    #[test]
    fn test_multiclass_confusion() {
        let mut cm = MultiClassConfusionMatrix::new(3);

        let preds = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[3, 3]),
            vec![
                0.7, 0.2, 0.1, // class 0
                0.1, 0.8, 0.1, // class 1
                0.2, 0.3, 0.5, // class 2
            ],
        ).unwrap();
        let targets = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[3, 3]),
            vec![
                1.0, 0.0, 0.0, // class 0
                0.0, 1.0, 0.0, // class 1
                0.0, 0.0, 1.0, // class 2
            ],
        ).unwrap();

        cm.update(&preds, &targets);
        assert!((cm.accuracy() - 1.0).abs() < 1e-6);
    }
}
