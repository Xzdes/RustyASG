// --- Файл: src/metrics/regression.rs ---

//! Метрики для задач регрессии.

use super::Metric;
use ndarray::ArrayD;

/// Mean Squared Error (MSE).
///
/// MSE = (1/n) * Σ(y_pred - y_true)²
#[derive(Debug, Clone, Default)]
pub struct MeanSquaredError {
    sum_squared_error: f64,
    count: usize,
}

impl MeanSquaredError {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Metric for MeanSquaredError {
    type Prediction = ArrayD<f32>;
    type Target = ArrayD<f32>;
    type Output = f64;

    fn update(&mut self, predictions: &Self::Prediction, targets: &Self::Target) {
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let diff = *pred as f64 - *target as f64;
            self.sum_squared_error += diff * diff;
            self.count += 1;
        }
    }

    fn compute(&self) -> Self::Output {
        if self.count == 0 {
            0.0
        } else {
            self.sum_squared_error / self.count as f64
        }
    }

    fn reset(&mut self) {
        self.sum_squared_error = 0.0;
        self.count = 0;
    }

    fn name(&self) -> &str {
        "MSE"
    }
}

/// Root Mean Squared Error (RMSE).
///
/// RMSE = √MSE
#[derive(Debug, Clone, Default)]
pub struct RootMeanSquaredError {
    mse: MeanSquaredError,
}

impl RootMeanSquaredError {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Metric for RootMeanSquaredError {
    type Prediction = ArrayD<f32>;
    type Target = ArrayD<f32>;
    type Output = f64;

    fn update(&mut self, predictions: &Self::Prediction, targets: &Self::Target) {
        self.mse.update(predictions, targets);
    }

    fn compute(&self) -> Self::Output {
        self.mse.compute().sqrt()
    }

    fn reset(&mut self) {
        self.mse.reset();
    }

    fn name(&self) -> &str {
        "RMSE"
    }
}

/// Mean Absolute Error (MAE).
///
/// MAE = (1/n) * Σ|y_pred - y_true|
#[derive(Debug, Clone, Default)]
pub struct MeanAbsoluteError {
    sum_absolute_error: f64,
    count: usize,
}

impl MeanAbsoluteError {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Metric for MeanAbsoluteError {
    type Prediction = ArrayD<f32>;
    type Target = ArrayD<f32>;
    type Output = f64;

    fn update(&mut self, predictions: &Self::Prediction, targets: &Self::Target) {
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let diff = (*pred as f64 - *target as f64).abs();
            self.sum_absolute_error += diff;
            self.count += 1;
        }
    }

    fn compute(&self) -> Self::Output {
        if self.count == 0 {
            0.0
        } else {
            self.sum_absolute_error / self.count as f64
        }
    }

    fn reset(&mut self) {
        self.sum_absolute_error = 0.0;
        self.count = 0;
    }

    fn name(&self) -> &str {
        "MAE"
    }
}

/// R-Squared (Coefficient of Determination).
///
/// R² = 1 - SS_res / SS_tot
/// где SS_res = Σ(y_true - y_pred)²
///     SS_tot = Σ(y_true - mean(y_true))²
#[derive(Debug, Clone, Default)]
pub struct RSquared {
    // Для Welford's online algorithm
    sum_squared_residuals: f64,
    sum_squared_total: f64,
    mean_target: f64,
    count: usize,
    targets_cache: Vec<f32>,
    predictions_cache: Vec<f32>,
}

impl RSquared {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Metric for RSquared {
    type Prediction = ArrayD<f32>;
    type Target = ArrayD<f32>;
    type Output = f64;

    fn update(&mut self, predictions: &Self::Prediction, targets: &Self::Target) {
        // Кэшируем данные для финального расчета
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            self.predictions_cache.push(*pred);
            self.targets_cache.push(*target);
        }
    }

    fn compute(&self) -> Self::Output {
        if self.targets_cache.is_empty() {
            return 0.0;
        }

        // Вычисляем среднее целевых значений
        let mean_target: f64 =
            self.targets_cache.iter().map(|&x| x as f64).sum::<f64>() / self.targets_cache.len() as f64;

        // SS_res = Σ(y_true - y_pred)²
        let ss_res: f64 = self
            .targets_cache
            .iter()
            .zip(self.predictions_cache.iter())
            .map(|(&t, &p)| {
                let diff = t as f64 - p as f64;
                diff * diff
            })
            .sum();

        // SS_tot = Σ(y_true - mean)²
        let ss_tot: f64 = self
            .targets_cache
            .iter()
            .map(|&t| {
                let diff = t as f64 - mean_target;
                diff * diff
            })
            .sum();

        if ss_tot == 0.0 {
            return 0.0;
        }

        1.0 - ss_res / ss_tot
    }

    fn reset(&mut self) {
        self.sum_squared_residuals = 0.0;
        self.sum_squared_total = 0.0;
        self.mean_target = 0.0;
        self.count = 0;
        self.targets_cache.clear();
        self.predictions_cache.clear();
    }

    fn name(&self) -> &str {
        "R²"
    }
}

/// Mean Absolute Percentage Error (MAPE).
///
/// MAPE = (1/n) * Σ|((y_true - y_pred) / y_true)| * 100
#[derive(Debug, Clone, Default)]
pub struct MeanAbsolutePercentageError {
    sum_ape: f64,
    count: usize,
    eps: f64,
}

impl MeanAbsolutePercentageError {
    pub fn new() -> Self {
        Self {
            sum_ape: 0.0,
            count: 0,
            eps: 1e-8,
        }
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }
}

impl Metric for MeanAbsolutePercentageError {
    type Prediction = ArrayD<f32>;
    type Target = ArrayD<f32>;
    type Output = f64;

    fn update(&mut self, predictions: &Self::Prediction, targets: &Self::Target) {
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let t = *target as f64;
            let p = *pred as f64;
            if t.abs() > self.eps {
                self.sum_ape += ((t - p) / t).abs();
                self.count += 1;
            }
        }
    }

    fn compute(&self) -> Self::Output {
        if self.count == 0 {
            0.0
        } else {
            (self.sum_ape / self.count as f64) * 100.0
        }
    }

    fn reset(&mut self) {
        self.sum_ape = 0.0;
        self.count = 0;
    }

    fn name(&self) -> &str {
        "MAPE"
    }
}

/// Explained Variance Score.
///
/// EVS = 1 - Var(y_true - y_pred) / Var(y_true)
#[derive(Debug, Clone, Default)]
pub struct ExplainedVariance {
    targets: Vec<f32>,
    predictions: Vec<f32>,
}

impl ExplainedVariance {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Metric for ExplainedVariance {
    type Prediction = ArrayD<f32>;
    type Target = ArrayD<f32>;
    type Output = f64;

    fn update(&mut self, predictions: &Self::Prediction, targets: &Self::Target) {
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            self.predictions.push(*pred);
            self.targets.push(*target);
        }
    }

    fn compute(&self) -> Self::Output {
        if self.targets.is_empty() {
            return 0.0;
        }

        let n = self.targets.len() as f64;

        // Вычисляем residuals
        let residuals: Vec<f64> = self
            .targets
            .iter()
            .zip(self.predictions.iter())
            .map(|(&t, &p)| t as f64 - p as f64)
            .collect();

        // Variance of residuals
        let mean_res = residuals.iter().sum::<f64>() / n;
        let var_res: f64 = residuals.iter().map(|&r| (r - mean_res).powi(2)).sum::<f64>() / n;

        // Variance of targets
        let mean_target = self.targets.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var_target: f64 = self
            .targets
            .iter()
            .map(|&t| (t as f64 - mean_target).powi(2))
            .sum::<f64>()
            / n;

        if var_target == 0.0 {
            return 0.0;
        }

        1.0 - var_res / var_target
    }

    fn reset(&mut self) {
        self.targets.clear();
        self.predictions.clear();
    }

    fn name(&self) -> &str {
        "ExplainedVariance"
    }
}

/// Max Error - максимальная абсолютная ошибка.
#[derive(Debug, Clone, Default)]
pub struct MaxError {
    max_error: f64,
}

impl MaxError {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Metric for MaxError {
    type Prediction = ArrayD<f32>;
    type Target = ArrayD<f32>;
    type Output = f64;

    fn update(&mut self, predictions: &Self::Prediction, targets: &Self::Target) {
        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let error = (*pred as f64 - *target as f64).abs();
            if error > self.max_error {
                self.max_error = error;
            }
        }
    }

    fn compute(&self) -> Self::Output {
        self.max_error
    }

    fn reset(&mut self) {
        self.max_error = 0.0;
    }

    fn name(&self) -> &str {
        "MaxError"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mse() {
        let mut mse = MeanSquaredError::new();

        let preds = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![1.0, 2.0, 3.0, 4.0],
        ).unwrap();
        let targets = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![1.0, 2.0, 3.0, 4.0],
        ).unwrap();

        mse.update(&preds, &targets);
        assert!((mse.compute() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_mae() {
        let mut mae = MeanAbsoluteError::new();

        let preds = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![1.0, 2.0, 3.0, 4.0],
        ).unwrap();
        let targets = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![2.0, 3.0, 4.0, 5.0],
        ).unwrap();

        mae.update(&preds, &targets);
        assert!((mae.compute() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_rmse() {
        let mut rmse = RootMeanSquaredError::new();

        let preds = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![1.0, 2.0, 3.0, 4.0],
        ).unwrap();
        let targets = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![2.0, 3.0, 4.0, 5.0],
        ).unwrap();

        rmse.update(&preds, &targets);
        assert!((rmse.compute() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_r_squared() {
        let mut r2 = RSquared::new();

        // Идеальное совпадение => R² = 1
        let preds = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![1.0, 2.0, 3.0, 4.0],
        ).unwrap();
        let targets = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![1.0, 2.0, 3.0, 4.0],
        ).unwrap();

        r2.update(&preds, &targets);
        assert!((r2.compute() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_max_error() {
        let mut max_err = MaxError::new();

        let preds = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![1.0, 2.0, 3.0, 10.0],
        ).unwrap();
        let targets = ArrayD::from_shape_vec(
            ndarray::IxDyn(&[4]),
            vec![1.0, 2.0, 3.0, 4.0],
        ).unwrap();

        max_err.update(&preds, &targets);
        assert!((max_err.compute() - 6.0).abs() < 1e-6);
    }
}
