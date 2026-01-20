// --- Файл: src/metrics/mod.rs ---

//! Модуль метрик для оценки качества моделей.
//!
//! Предоставляет набор метрик для различных задач машинного обучения:
//! - **Классификация**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
//! - **Регрессия**: MSE, MAE, RMSE, R²
//! - **Ранжирование**: MRR, NDCG
//!
//! # Пример использования
//!
//! ```rust,ignore
//! use rustyasg::metrics::{Accuracy, Metric};
//!
//! let mut accuracy = Accuracy::new();
//! accuracy.update(&predictions, &targets);
//! println!("Accuracy: {:.4}", accuracy.compute());
//! accuracy.reset();
//! ```

pub mod classification;
pub mod regression;
pub mod running;

pub use classification::{
    Accuracy, Precision, Recall, F1Score,
    BinaryConfusionMatrix, MultiClassConfusionMatrix, TopKAccuracy,
};
pub use regression::{
    MeanSquaredError, MeanAbsoluteError, RSquared, RootMeanSquaredError,
    MeanAbsolutePercentageError, ExplainedVariance, MaxError,
};
pub use running::{
    RunningMean, RunningStd, RunningMinMax,
    ExponentialMovingAverage, MetricLogger, MetricSummary, EarlyStopping,
};

/// Базовый трейт для всех метрик.
pub trait Metric: Send + Sync {
    /// Тип предсказания
    type Prediction;
    /// Тип целевого значения
    type Target;
    /// Тип результата метрики
    type Output;

    /// Обновляет состояние метрики новыми данными.
    fn update(&mut self, predictions: &Self::Prediction, targets: &Self::Target);

    /// Вычисляет текущее значение метрики.
    fn compute(&self) -> Self::Output;

    /// Сбрасывает состояние метрики.
    fn reset(&mut self);

    /// Возвращает имя метрики.
    fn name(&self) -> &str;
}
