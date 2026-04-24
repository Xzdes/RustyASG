//! Utilities for tracking statistics online.
//!
//! Uses numerically stable incremental algorithms (Welford's method) so
//! `mean` and `std` can be computed without keeping the whole history in
//! memory.

use std::collections::HashMap;

/// Online running mean.
///
/// Uses an incremental update that avoids the precision loss of repeatedly
/// summing floats.
#[derive(Debug, Clone, Default)]
pub struct RunningMean {
    mean: f64,
    count: usize,
}

impl RunningMean {
    /// Creates a new `RunningMean`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a single value.
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
    }

    /// Adds a batch of values.
    pub fn update_batch(&mut self, values: &[f64]) {
        for &v in values {
            self.update(v);
        }
    }

    /// Returns the current mean.
    pub fn compute(&self) -> f64 {
        self.mean
    }

    /// Returns the number of values seen so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Resets the accumulator.
    pub fn reset(&mut self) {
        self.mean = 0.0;
        self.count = 0;
    }
}

/// Online running mean + standard deviation (Welford's algorithm).
///
/// Computes mean, variance and std without storing individual values.
#[derive(Debug, Clone, Default)]
pub struct RunningStd {
    count: usize,
    mean: f64,
    m2: f64, // Sum of squares of differences from the mean.
}

impl RunningStd {
    /// Creates a new `RunningStd`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a single value using Welford's online update.
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Adds a batch of values.
    pub fn update_batch(&mut self, values: &[f64]) {
        for &v in values {
            self.update(v);
        }
    }

    /// Returns the current mean.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Returns the sample variance (`n - 1` in the denominator).
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    /// Returns the population variance (`n` in the denominator).
    pub fn population_variance(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.m2 / self.count as f64
        }
    }

    /// Returns the sample standard deviation.
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Returns the population standard deviation.
    pub fn population_std(&self) -> f64 {
        self.population_variance().sqrt()
    }

    /// Returns the number of values seen so far.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Resets the accumulator.
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }
}

/// Online running min / max tracker.
#[derive(Debug, Clone)]
pub struct RunningMinMax {
    min: f64,
    max: f64,
    count: usize,
}

impl RunningMinMax {
    /// Creates a new `RunningMinMax`.
    pub fn new() -> Self {
        Self {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            count: 0,
        }
    }

    /// Adds a value.
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    /// Returns the minimum, or `None` if no values have been added.
    pub fn min(&self) -> Option<f64> {
        if self.count > 0 {
            Some(self.min)
        } else {
            None
        }
    }

    /// Returns the maximum, or `None` if no values have been added.
    pub fn max(&self) -> Option<f64> {
        if self.count > 0 {
            Some(self.max)
        } else {
            None
        }
    }

    /// Returns `max - min`.
    pub fn range(&self) -> Option<f64> {
        if self.count > 0 {
            Some(self.max - self.min)
        } else {
            None
        }
    }

    /// Resets the accumulator.
    pub fn reset(&mut self) {
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
        self.count = 0;
    }
}

impl Default for RunningMinMax {
    fn default() -> Self {
        Self::new()
    }
}

/// Exponential moving average (EMA).
#[derive(Debug, Clone)]
pub struct ExponentialMovingAverage {
    alpha: f64,
    value: Option<f64>,
}

impl ExponentialMovingAverage {
    /// Creates an EMA with the given smoothing factor.
    ///
    /// `alpha` must be in `(0, 1]`. Smaller values produce smoother curves.
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
        Self { alpha, value: None }
    }

    /// Creates an EMA with `alpha` derived from a span:
    /// `alpha = 2 / (span + 1)`.
    pub fn from_span(span: usize) -> Self {
        let alpha = 2.0 / (span as f64 + 1.0);
        Self::new(alpha)
    }

    /// Adds a value.
    pub fn update(&mut self, value: f64) {
        self.value = Some(match self.value {
            Some(prev) => self.alpha * value + (1.0 - self.alpha) * prev,
            None => value,
        });
    }

    /// Returns the current EMA value.
    pub fn compute(&self) -> Option<f64> {
        self.value
    }

    /// Resets the accumulator.
    pub fn reset(&mut self) {
        self.value = None;
    }
}

/// Metric logger for tracking multiple metrics across training.
#[derive(Debug, Clone, Default)]
pub struct MetricLogger {
    /// Per-epoch history of metric values.
    history: HashMap<String, Vec<f64>>,
    /// Running means for the current epoch.
    current: HashMap<String, RunningMean>,
    /// Best value seen per metric as `(value, epoch)`.
    best: HashMap<String, (f64, usize)>,
    /// Current epoch counter.
    epoch: usize,
}

impl MetricLogger {
    /// Creates a new `MetricLogger`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Logs a single metric value for the current epoch.
    pub fn log(&mut self, name: &str, value: f64) {
        self.current
            .entry(name.to_string())
            .or_default()
            .update(value);
    }

    /// Logs several metrics at once.
    pub fn log_dict(&mut self, metrics: &HashMap<String, f64>) {
        for (name, &value) in metrics {
            self.log(name, value);
        }
    }

    /// Finalises the current epoch: snapshots averages to `history` and
    /// refreshes the `best` record (minimisation is assumed).
    pub fn end_epoch(&mut self) {
        for (name, running_mean) in &self.current {
            let value = running_mean.compute();

            // Append to history.
            self.history.entry(name.clone()).or_default().push(value);

            // Update the best value (minimisation is assumed).
            let is_better = match self.best.get(name) {
                Some((best_val, _)) => value < *best_val,
                None => true,
            };
            if is_better {
                self.best.insert(name.clone(), (value, self.epoch));
            }
        }

        // Clear per-epoch accumulators.
        self.current.clear();
        self.epoch += 1;
    }

    /// Returns the full history for a metric.
    pub fn get_history(&self, name: &str) -> Option<&Vec<f64>> {
        self.history.get(name)
    }

    /// Returns the best value and epoch for a metric.
    pub fn get_best(&self, name: &str) -> Option<(f64, usize)> {
        self.best.get(name).copied()
    }

    /// Returns the most recent epoch's value for a metric.
    pub fn get_last(&self, name: &str) -> Option<f64> {
        self.history.get(name).and_then(|h| h.last().copied())
    }

    /// Returns the running mean inside the current (unfinished) epoch.
    pub fn get_current_mean(&self, name: &str) -> Option<f64> {
        self.current.get(name).map(|m| m.compute())
    }

    /// Returns the current epoch number.
    pub fn current_epoch(&self) -> usize {
        self.epoch
    }

    /// Returns the names of every metric that has ever been logged.
    pub fn metric_names(&self) -> Vec<&String> {
        self.history.keys().collect()
    }

    /// Returns a `MetricSummary` per metric (last / best / mean / epochs).
    pub fn summary(&self) -> HashMap<String, MetricSummary> {
        let mut result = HashMap::new();

        for (name, history) in &self.history {
            let last = history.last().copied();
            let best = self.best.get(name).copied();
            let mean = if history.is_empty() {
                None
            } else {
                Some(history.iter().sum::<f64>() / history.len() as f64)
            };

            result.insert(
                name.clone(),
                MetricSummary {
                    last,
                    best_value: best.map(|(v, _)| v),
                    best_epoch: best.map(|(_, e)| e),
                    mean,
                    epochs: history.len(),
                },
            );
        }

        result
    }

    /// Formats the in-progress epoch as `"loss: 0.123 | accuracy: 0.987"`.
    pub fn format_current(&self) -> String {
        let mut parts: Vec<String> = self
            .current
            .iter()
            .map(|(name, mean)| format!("{}: {:.6}", name, mean.compute()))
            .collect();
        parts.sort();
        parts.join(" | ")
    }

    /// Formats a finished epoch for printing, marking the best epoch with a
    /// trailing `" *"`.
    pub fn format_epoch(&self, epoch: usize) -> String {
        let mut parts: Vec<String> = Vec::new();

        for (name, history) in &self.history {
            if let Some(&value) = history.get(epoch) {
                let best_marker = match self.best.get(name) {
                    Some((_, best_epoch)) if *best_epoch == epoch => " *",
                    _ => "",
                };
                parts.push(format!("{}: {:.6}{}", name, value, best_marker));
            }
        }

        parts.sort();
        format!("Epoch {}: {}", epoch, parts.join(" | "))
    }

    /// Resets the logger to a clean state.
    pub fn reset(&mut self) {
        self.history.clear();
        self.current.clear();
        self.best.clear();
        self.epoch = 0;
    }
}

/// Snapshot of a single metric's history.
#[derive(Debug, Clone)]
pub struct MetricSummary {
    /// Last recorded value.
    pub last: Option<f64>,
    /// Best value ever recorded.
    pub best_value: Option<f64>,
    /// Epoch at which the best value was recorded.
    pub best_epoch: Option<usize>,
    /// Mean over all recorded values.
    pub mean: Option<f64>,
    /// Total number of epochs recorded.
    pub epochs: usize,
}

/// Early-stopping tracker.
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    /// Name of the metric being watched.
    metric_name: String,
    /// Whether we minimise or maximise the metric.
    mode: EarlyStoppingMode,
    /// Number of epochs without improvement before stopping.
    patience: usize,
    /// Minimum change to count as an improvement.
    min_delta: f64,
    /// Current count of epochs without improvement.
    counter: usize,
    /// Best value seen so far.
    best_value: Option<f64>,
    /// Epoch at which the best value was seen.
    best_epoch: usize,
    /// Whether training should stop now.
    should_stop: bool,
}

/// Direction for early stopping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlyStoppingMode {
    /// Minimise the metric (e.g. loss).
    Min,
    /// Maximise the metric (e.g. accuracy).
    Max,
}

impl EarlyStopping {
    /// Creates an `EarlyStopping` that minimises `metric_name`.
    pub fn minimize(metric_name: &str, patience: usize) -> Self {
        Self {
            metric_name: metric_name.to_string(),
            mode: EarlyStoppingMode::Min,
            patience,
            min_delta: 0.0,
            counter: 0,
            best_value: None,
            best_epoch: 0,
            should_stop: false,
        }
    }

    /// Creates an `EarlyStopping` that maximises `metric_name`.
    pub fn maximize(metric_name: &str, patience: usize) -> Self {
        Self {
            metric_name: metric_name.to_string(),
            mode: EarlyStoppingMode::Max,
            patience,
            min_delta: 0.0,
            counter: 0,
            best_value: None,
            best_epoch: 0,
            should_stop: false,
        }
    }

    /// Sets the minimum delta that counts as an improvement.
    pub fn with_min_delta(mut self, min_delta: f64) -> Self {
        self.min_delta = min_delta;
        self
    }

    /// Records a metric value for `epoch` and returns whether training
    /// should stop.
    pub fn check(&mut self, value: f64, epoch: usize) -> bool {
        let is_better = match (self.mode, self.best_value) {
            (EarlyStoppingMode::Min, Some(best)) => value < best - self.min_delta,
            (EarlyStoppingMode::Max, Some(best)) => value > best + self.min_delta,
            (_, None) => true,
        };

        if is_better {
            self.best_value = Some(value);
            self.best_epoch = epoch;
            self.counter = 0;
        } else {
            self.counter += 1;
            if self.counter >= self.patience {
                self.should_stop = true;
            }
        }

        self.should_stop
    }

    /// Returns the name of the tracked metric.
    pub fn metric_name(&self) -> &str {
        &self.metric_name
    }

    /// Returns the best value seen so far.
    pub fn best_value(&self) -> Option<f64> {
        self.best_value
    }

    /// Returns the epoch at which the best value was seen.
    pub fn best_epoch(&self) -> usize {
        self.best_epoch
    }

    /// Returns whether training should stop now.
    pub fn should_stop(&self) -> bool {
        self.should_stop
    }

    /// Returns the current patience counter (epochs without improvement).
    pub fn counter(&self) -> usize {
        self.counter
    }

    /// Resets the tracker to a clean state.
    pub fn reset(&mut self) {
        self.counter = 0;
        self.best_value = None;
        self.best_epoch = 0;
        self.should_stop = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_running_mean() {
        let mut rm = RunningMean::new();
        rm.update(1.0);
        rm.update(2.0);
        rm.update(3.0);
        rm.update(4.0);
        rm.update(5.0);

        assert!((rm.compute() - 3.0).abs() < 1e-10);
        assert_eq!(rm.count(), 5);
    }

    #[test]
    fn test_running_std() {
        let mut rs = RunningStd::new();
        rs.update_batch(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);

        assert!((rs.mean() - 5.0).abs() < 1e-10);
        assert!((rs.population_variance() - 4.0).abs() < 1e-10);
        assert!((rs.population_std() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_running_min_max() {
        let mut rmm = RunningMinMax::new();
        rmm.update(5.0);
        rmm.update(2.0);
        rmm.update(8.0);
        rmm.update(1.0);
        rmm.update(9.0);

        assert_eq!(rmm.min(), Some(1.0));
        assert_eq!(rmm.max(), Some(9.0));
        assert_eq!(rmm.range(), Some(8.0));
    }

    #[test]
    fn test_ema() {
        let mut ema = ExponentialMovingAverage::new(0.5);
        ema.update(10.0);
        assert_eq!(ema.compute(), Some(10.0));

        ema.update(20.0);
        assert_eq!(ema.compute(), Some(15.0)); // 0.5 * 20 + 0.5 * 10

        ema.update(30.0);
        assert_eq!(ema.compute(), Some(22.5)); // 0.5 * 30 + 0.5 * 15
    }

    #[test]
    fn test_metric_logger() {
        let mut logger = MetricLogger::new();

        // Epoch 0.
        logger.log("loss", 1.0);
        logger.log("loss", 0.8);
        logger.log("accuracy", 0.7);
        logger.end_epoch();

        // Epoch 1.
        logger.log("loss", 0.6);
        logger.log("loss", 0.5);
        logger.log("accuracy", 0.85);
        logger.end_epoch();

        assert_eq!(logger.current_epoch(), 2);
        assert!((logger.get_last("loss").unwrap() - 0.55).abs() < 1e-10);
        assert!((logger.get_best("loss").unwrap().0 - 0.55).abs() < 1e-10);
        assert_eq!(logger.get_best("loss").unwrap().1, 1);
    }

    #[test]
    fn test_early_stopping() {
        let mut es = EarlyStopping::minimize("loss", 3);

        // Improvements.
        assert!(!es.check(1.0, 0));
        assert!(!es.check(0.9, 1));
        assert!(!es.check(0.8, 2));

        // No improvement.
        assert!(!es.check(0.85, 3)); // counter = 1
        assert!(!es.check(0.9, 4)); // counter = 2
        assert!(es.check(0.95, 5)); // counter = 3, should_stop = true

        assert!(es.should_stop());
        assert_eq!(es.best_epoch(), 2);
        assert!((es.best_value().unwrap() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_early_stopping_maximize() {
        let mut es = EarlyStopping::maximize("accuracy", 2);

        assert!(!es.check(0.7, 0));
        assert!(!es.check(0.8, 1));
        assert!(!es.check(0.75, 2)); // counter = 1
        assert!(es.check(0.76, 3)); // counter = 2, should_stop

        assert_eq!(es.best_epoch(), 1);
    }
}
