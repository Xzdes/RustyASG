//! Модуль, содержащий реализации функций потерь в графовой парадигме.
//!
//! Функции потерь здесь - это обычные Rust-функции, которые принимают
//! символьные `Tensor`-дескрипторы и добавляют в граф узлы, необходимые
//! для вычисления значения ошибки.

use crate::tensor::Tensor;
use ndarray::arr0;

/// Вычисляет символьный граф для Среднеквадратичной Ошибки (MSE).
///
/// Формула: `MSE = sum((y_pred - y_true)^2)`.
///
/// # Аргументы
///
/// * `y_pred` - Символьный `Tensor` с предсказаниями модели.
/// * `y_true` - Символьный `Tensor` с истинными значениями.
///
/// # Возвращает
///
/// Новый символьный `Tensor`, представляющий скалярный узел (loss) в графе.
pub fn mse_loss(y_pred: &Tensor, y_true: &Tensor) -> Tensor {
    // (y_pred - y_true)
    let error = y_pred - y_true;

    // error^2
    // Для возведения в квадрат нам нужен узел-константа со значением 2.0
    let power_context = &error.context;
    let two = Tensor::new_literal(
        power_context,
        arr0(2.0f32).into_dyn(),
        "two_literal",
    );
    
    // Используем метод .pow()
    let squared_error = error.pow(&two);

    // sum(squared_error)
    let loss = squared_error.sum();
    loss
}