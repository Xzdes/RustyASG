//! Модуль, определяющий типы ошибок для библиотеки RustyGradients.

use thiserror::Error;

/// Основной тип `Result` для этой библиотеки.
pub type Result<T> = std::result::Result<T, RustyGradientsError>;

/// Перечисление всех возможных ошибок, которые могут возникнуть при работе с библиотекой.
///
/// Использование `thiserror` позволяет нам легко создавать информативные
/// и структурированные сообщения об ошибках.
#[derive(Error, Debug)]
pub enum RustyGradientsError {
    /// Возникает при несовместимости форм (shapes) тензоров в операции.
    /// Например, при попытке сложить тензоры форм `[2, 3]` и `[3, 2]`.
    #[error("Incompatible shapes for operation: {0}")]
    ShapeError(String),

    /// Возникает, когда операция ожидает тензор определенной размерности (ndim),
    /// но получает другую.
    #[error("Operation requires a tensor with {expected} dimensions, but got {actual}")]
    DimensionError { expected: usize, actual: usize },

    /// Возникает при попытке создать тензор из среза с неверной формой.
    #[error("Cannot create tensor with shape {shape:?} from slice of length {slice_len}")]
    ShapeCreationError {
        shape: Vec<usize>,
        slice_len: usize,
    },

    /// Возникает, когда операция требует определенной длины оси, но получает другую.
    #[error("Invalid axis dimension: expected {expected} for axis {axis}, but got {actual}")]
    AxisDimensionError {
        axis: usize,
        expected: usize,
        actual: usize,
    },

    /// Обобщенная ошибка для различных невалидных входных данных.
    #[error("Invalid input: {0}")]
    InvalidInput(String),
}