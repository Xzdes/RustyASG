// --- Файл: src/metrics/running.rs ---

//! Утилиты для отслеживания статистик в реальном времени.
//!
//! Использует онлайн-алгоритмы (Welford) для численно стабильного
//! вычисления mean и std без хранения всех значений.

use std::collections::HashMap;

/// Онлайн вычисление среднего значения.
///
/// Использует инкрементальный алгоритм для численной стабильности.
#[derive(Debug, Clone, Default)]
pub struct RunningMean {
    mean: f64,
    count: usize,
}

impl RunningMean {
    /// Создает новый RunningMean.
    pub fn new() -> Self {
        Self::default()
    }

    /// Добавляет значение.
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
    }

    /// Добавляет несколько значений.
    pub fn update_batch(&mut self, values: &[f64]) {
        for &v in values {
            self.update(v);
        }
    }

    /// Возвращает текущее среднее.
    pub fn compute(&self) -> f64 {
        self.mean
    }

    /// Возвращает количество добавленных значений.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Сбрасывает состояние.
    pub fn reset(&mut self) {
        self.mean = 0.0;
        self.count = 0;
    }
}

/// Онлайн вычисление стандартного отклонения (Welford's algorithm).
///
/// Вычисляет mean, variance и std без хранения всех значений.
#[derive(Debug, Clone, Default)]
pub struct RunningStd {
    count: usize,
    mean: f64,
    m2: f64, // Sum of squares of differences from mean
}

impl RunningStd {
    /// Создает новый RunningStd.
    pub fn new() -> Self {
        Self::default()
    }

    /// Добавляет значение (Welford's online algorithm).
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Добавляет несколько значений.
    pub fn update_batch(&mut self, values: &[f64]) {
        for &v in values {
            self.update(v);
        }
    }

    /// Возвращает среднее значение.
    pub fn mean(&self) -> f64 {
        self.mean
    }

    /// Возвращает дисперсию (sample variance).
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    /// Возвращает population variance.
    pub fn population_variance(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.m2 / self.count as f64
        }
    }

    /// Возвращает стандартное отклонение (sample std).
    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Возвращает population std.
    pub fn population_std(&self) -> f64 {
        self.population_variance().sqrt()
    }

    /// Возвращает количество значений.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Сбрасывает состояние.
    pub fn reset(&mut self) {
        self.count = 0;
        self.mean = 0.0;
        self.m2 = 0.0;
    }
}

/// Онлайн вычисление минимума и максимума.
#[derive(Debug, Clone)]
pub struct RunningMinMax {
    min: f64,
    max: f64,
    count: usize,
}

impl RunningMinMax {
    /// Создает новый RunningMinMax.
    pub fn new() -> Self {
        Self {
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            count: 0,
        }
    }

    /// Добавляет значение.
    pub fn update(&mut self, value: f64) {
        self.count += 1;
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }
    }

    /// Возвращает минимум.
    pub fn min(&self) -> Option<f64> {
        if self.count > 0 {
            Some(self.min)
        } else {
            None
        }
    }

    /// Возвращает максимум.
    pub fn max(&self) -> Option<f64> {
        if self.count > 0 {
            Some(self.max)
        } else {
            None
        }
    }

    /// Возвращает range (max - min).
    pub fn range(&self) -> Option<f64> {
        if self.count > 0 {
            Some(self.max - self.min)
        } else {
            None
        }
    }

    /// Сбрасывает состояние.
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

/// Экспоненциальное скользящее среднее (EMA).
#[derive(Debug, Clone)]
pub struct ExponentialMovingAverage {
    alpha: f64,
    value: Option<f64>,
}

impl ExponentialMovingAverage {
    /// Создает EMA с заданным коэффициентом сглаживания.
    ///
    /// alpha должен быть в диапазоне (0, 1].
    /// Меньшие значения дают более гладкое среднее.
    pub fn new(alpha: f64) -> Self {
        assert!(alpha > 0.0 && alpha <= 1.0, "alpha must be in (0, 1]");
        Self { alpha, value: None }
    }

    /// Создает EMA с коэффициентом, вычисленным из span.
    ///
    /// alpha = 2 / (span + 1)
    pub fn from_span(span: usize) -> Self {
        let alpha = 2.0 / (span as f64 + 1.0);
        Self::new(alpha)
    }

    /// Добавляет значение.
    pub fn update(&mut self, value: f64) {
        self.value = Some(match self.value {
            Some(prev) => self.alpha * value + (1.0 - self.alpha) * prev,
            None => value,
        });
    }

    /// Возвращает текущее EMA.
    pub fn compute(&self) -> Option<f64> {
        self.value
    }

    /// Сбрасывает состояние.
    pub fn reset(&mut self) {
        self.value = None;
    }
}

/// Логгер метрик для отслеживания множества метрик во время обучения.
#[derive(Debug, Clone, Default)]
pub struct MetricLogger {
    /// История значений метрик по эпохам
    history: HashMap<String, Vec<f64>>,
    /// Текущие значения (для текущей эпохи)
    current: HashMap<String, RunningMean>,
    /// Лучшие значения
    best: HashMap<String, (f64, usize)>, // (value, epoch)
    /// Текущая эпоха
    epoch: usize,
}

impl MetricLogger {
    /// Создает новый MetricLogger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Логирует значение метрики.
    pub fn log(&mut self, name: &str, value: f64) {
        self.current
            .entry(name.to_string())
            .or_insert_with(RunningMean::new)
            .update(value);
    }

    /// Логирует несколько метрик сразу.
    pub fn log_dict(&mut self, metrics: &HashMap<String, f64>) {
        for (name, &value) in metrics {
            self.log(name, value);
        }
    }

    /// Завершает эпоху: сохраняет средние значения и обновляет best.
    pub fn end_epoch(&mut self) {
        for (name, running_mean) in &self.current {
            let value = running_mean.compute();

            // Добавляем в историю
            self.history
                .entry(name.clone())
                .or_insert_with(Vec::new)
                .push(value);

            // Обновляем лучшее значение (предполагаем минимизацию)
            let is_better = match self.best.get(name) {
                Some((best_val, _)) => value < *best_val,
                None => true,
            };
            if is_better {
                self.best.insert(name.clone(), (value, self.epoch));
            }
        }

        // Очищаем текущие значения
        self.current.clear();
        self.epoch += 1;
    }

    /// Возвращает историю метрики.
    pub fn get_history(&self, name: &str) -> Option<&Vec<f64>> {
        self.history.get(name)
    }

    /// Возвращает лучшее значение метрики.
    pub fn get_best(&self, name: &str) -> Option<(f64, usize)> {
        self.best.get(name).copied()
    }

    /// Возвращает последнее значение метрики.
    pub fn get_last(&self, name: &str) -> Option<f64> {
        self.history.get(name).and_then(|h| h.last().copied())
    }

    /// Возвращает текущее среднее (в рамках эпохи).
    pub fn get_current_mean(&self, name: &str) -> Option<f64> {
        self.current.get(name).map(|m| m.compute())
    }

    /// Возвращает текущую эпоху.
    pub fn current_epoch(&self) -> usize {
        self.epoch
    }

    /// Возвращает все имена метрик.
    pub fn metric_names(&self) -> Vec<&String> {
        self.history.keys().collect()
    }

    /// Возвращает сводку по всем метрикам.
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

    /// Форматирует текущее состояние для вывода.
    pub fn format_current(&self) -> String {
        let mut parts: Vec<String> = self
            .current
            .iter()
            .map(|(name, mean)| format!("{}: {:.6}", name, mean.compute()))
            .collect();
        parts.sort();
        parts.join(" | ")
    }

    /// Форматирует результаты эпохи.
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

    /// Сбрасывает состояние логгера.
    pub fn reset(&mut self) {
        self.history.clear();
        self.current.clear();
        self.best.clear();
        self.epoch = 0;
    }
}

/// Сводка по метрике.
#[derive(Debug, Clone)]
pub struct MetricSummary {
    /// Последнее значение
    pub last: Option<f64>,
    /// Лучшее значение
    pub best_value: Option<f64>,
    /// Эпоха лучшего значения
    pub best_epoch: Option<usize>,
    /// Среднее по всем эпохам
    pub mean: Option<f64>,
    /// Количество эпох
    pub epochs: usize,
}

/// Трекер для early stopping.
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    /// Метрика для отслеживания
    metric_name: String,
    /// Режим: "min" или "max"
    mode: EarlyStoppingMode,
    /// Количество эпох без улучшения до остановки
    patience: usize,
    /// Минимальное улучшение для считания прогрессом
    min_delta: f64,
    /// Текущий счетчик эпох без улучшения
    counter: usize,
    /// Лучшее значение
    best_value: Option<f64>,
    /// Эпоха лучшего значения
    best_epoch: usize,
    /// Флаг остановки
    should_stop: bool,
}

/// Режим early stopping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EarlyStoppingMode {
    /// Минимизация метрики (например, loss)
    Min,
    /// Максимизация метрики (например, accuracy)
    Max,
}

impl EarlyStopping {
    /// Создает новый EarlyStopping для минимизации.
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

    /// Создает новый EarlyStopping для максимизации.
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

    /// Устанавливает минимальное улучшение.
    pub fn with_min_delta(mut self, min_delta: f64) -> Self {
        self.min_delta = min_delta;
        self
    }

    /// Проверяет значение метрики и обновляет состояние.
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

    /// Возвращает имя отслеживаемой метрики.
    pub fn metric_name(&self) -> &str {
        &self.metric_name
    }

    /// Возвращает лучшее значение.
    pub fn best_value(&self) -> Option<f64> {
        self.best_value
    }

    /// Возвращает эпоху лучшего значения.
    pub fn best_epoch(&self) -> usize {
        self.best_epoch
    }

    /// Проверяет, нужно ли остановиться.
    pub fn should_stop(&self) -> bool {
        self.should_stop
    }

    /// Возвращает текущий счетчик терпения.
    pub fn counter(&self) -> usize {
        self.counter
    }

    /// Сбрасывает состояние.
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

        // Эпоха 0
        logger.log("loss", 1.0);
        logger.log("loss", 0.8);
        logger.log("accuracy", 0.7);
        logger.end_epoch();

        // Эпоха 1
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

        // Улучшение
        assert!(!es.check(1.0, 0));
        assert!(!es.check(0.9, 1));
        assert!(!es.check(0.8, 2));

        // Без улучшения
        assert!(!es.check(0.85, 3)); // counter = 1
        assert!(!es.check(0.9, 4));  // counter = 2
        assert!(es.check(0.95, 5));  // counter = 3, should_stop = true

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
        assert!(es.check(0.76, 3));  // counter = 2, should_stop

        assert_eq!(es.best_epoch(), 1);
    }
}
