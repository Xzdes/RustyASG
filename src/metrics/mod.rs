//! Metrics module for evaluating model quality.
//!
//! Provides a set of metrics for various machine-learning tasks:
//! - **Classification**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
//! - **Regression**: MSE, MAE, RMSE, R²
//! - **Ranking**: MRR, NDCG
//!
//! # Example
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
    Accuracy, BinaryConfusionMatrix, F1Score, MultiClassConfusionMatrix, Precision, Recall,
    TopKAccuracy,
};
pub use regression::{
    ExplainedVariance, MaxError, MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredError,
    RSquared, RootMeanSquaredError,
};
pub use running::{
    EarlyStopping, ExponentialMovingAverage, MetricLogger, MetricSummary, RunningMean,
    RunningMinMax, RunningStd,
};

/// Base trait for all metrics.
pub trait Metric: Send + Sync {
    /// Prediction type.
    type Prediction;
    /// Target value type.
    type Target;
    /// Metric output type.
    type Output;

    /// Updates the metric state with new data.
    fn update(&mut self, predictions: &Self::Prediction, targets: &Self::Target);

    /// Computes the current metric value.
    fn compute(&self) -> Self::Output;

    /// Resets the metric state.
    fn reset(&mut self);

    /// Returns the metric name.
    fn name(&self) -> &str;
}
