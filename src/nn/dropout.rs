//! Dropout слой для регуляризации.
//!
//! Реализует стандартный Dropout, который случайно обнуляет
//! элементы тензора во время обучения для предотвращения переобучения.

use crate::nn::Module;
use crate::tensor::Tensor;

/// Слой Dropout для регуляризации.
///
/// Во время обучения случайно обнуляет элементы с вероятностью `p`,
/// масштабируя остальные на 1/(1-p) для сохранения математического ожидания.
///
/// Во время inference (eval mode) просто пропускает вход без изменений.
///
/// # Пример
/// ```ignore
/// let dropout = Dropout::new(0.5); // 50% вероятность обнуления
/// dropout.eval(); // Отключить dropout для inference
/// ```
pub struct Dropout {
    /// Вероятность обнуления (0.0 - 1.0)
    pub p: f32,
    /// Флаг режима обучения
    pub training: bool,
}

impl Dropout {
    /// Создаёт новый слой Dropout.
    ///
    /// # Аргументы
    /// * `p` - Вероятность обнуления элемента (рекомендуется 0.1-0.5)
    ///
    /// # Panics
    /// Паникует если `p` не в диапазоне [0, 1)
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1), got {}",
            p
        );
        Self { p, training: true }
    }

    /// Устанавливает режим обучения.
    pub fn train(&mut self) {
        self.training = true;
    }

    /// Устанавливает режим inference.
    pub fn eval(&mut self) {
        self.training = false;
    }

    /// Проверяет, включен ли dropout.
    pub fn is_training(&self) -> bool {
        self.training
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl Module for Dropout {
    /// Прямой проход Dropout.
    ///
    /// **Примечание**: В текущей графовой архитектуре Dropout не применяется
    /// на этапе построения графа, так как случайность должна быть привнесена
    /// во время выполнения. Этот метод просто возвращает вход без изменений.
    ///
    /// Для полной поддержки Dropout необходимо:
    /// 1. Добавить NodeType::Dropout в ASG
    /// 2. Реализовать генерацию маски в runtime
    /// 3. Передавать флаг training в backend
    fn forward(&self, x: &Tensor) -> Tensor {
        // TODO: Реализовать через специальный узел Dropout в ASG
        // Пока что просто возвращаем вход
        x.clone()
    }

    fn parameters(&self) -> Vec<Tensor> {
        // Dropout не имеет обучаемых параметров
        Vec::new()
    }
}

/// Spatial Dropout для сверточных сетей.
///
/// В отличие от стандартного Dropout, обнуляет целые каналы (feature maps),
/// что более эффективно для сверточных архитектур.
pub struct SpatialDropout {
    /// Вероятность обнуления канала
    pub p: f32,
    /// Флаг режима обучения
    pub training: bool,
}

impl SpatialDropout {
    /// Создаёт новый слой SpatialDropout.
    pub fn new(p: f32) -> Self {
        assert!(
            (0.0..1.0).contains(&p),
            "Dropout probability must be in [0, 1), got {}",
            p
        );
        Self { p, training: true }
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }
}

impl Module for SpatialDropout {
    fn forward(&self, x: &Tensor) -> Tensor {
        // TODO: Реализовать через специальный узел
        x.clone()
    }

    fn parameters(&self) -> Vec<Tensor> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dropout_creation() {
        let dropout = Dropout::new(0.5);
        assert_eq!(dropout.p, 0.5);
        assert!(dropout.training);
    }

    #[test]
    fn test_dropout_modes() {
        let mut dropout = Dropout::new(0.3);

        dropout.eval();
        assert!(!dropout.training);

        dropout.train();
        assert!(dropout.training);
    }

    #[test]
    #[should_panic(expected = "Dropout probability must be in [0, 1)")]
    fn test_dropout_invalid_p() {
        Dropout::new(1.5);
    }
}
