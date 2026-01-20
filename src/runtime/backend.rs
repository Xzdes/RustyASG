//! Модуль, определяющий абстрактный интерфейс (трейт) для исполнительных бэкендов.

use crate::asg::{Asg, AsgId, NodeId, Value};
use std::collections::HashMap;
use thiserror::Error;

/// Ошибки, которые могут возникнуть во время выполнения (интерпретации) графа.
/// Этот тип ошибок является общим для всех бэкендов.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum RuntimeError {
    #[error("Узел с ID {0} не найден в графе {1}. Проверьте, что граф был корректно построен и все узлы существуют.")]
    NodeNotFound(NodeId, AsgId),

    #[error("Граф с ID {0} не найден в контексте выполнения. Убедитесь, что граф был зарегистрирован перед выполнением.")]
    GraphNotFound(AsgId),

    #[error("Несоответствие типов: операция ожидала {expected}, но получила {actual}. Проверьте типы входных данных.")]
    TypeError { expected: String, actual: String },

    #[error("Ошибка формы тензора: {0}. Проверьте размерности входных тензоров.")]
    ShapeError(String),

    #[error("Отсутствует значение для входа '{0}' (узел ID: {1}). Добавьте это значение в initial_data при вызове backend.run().")]
    MissingInput(String, NodeId),

    #[error("Отсутствует значение для параметра '{0}' (узел ID: {1}). Инициализируйте параметр перед выполнением графа.")]
    MissingParameter(String, NodeId),

    #[error("Операция '{0}' не реализована в текущем бэкенде. Рассмотрите использование альтернативной операции или реализуйте поддержку.")]
    UnimplementedOperation(String),

    #[error("Ошибка вычисления: {0}")]
    ComputationError(String),

    #[error("Ошибка памяти: {0}")]
    MemoryError(String),
}

/// Кэш для хранения уже вычисленных значений узлов.
/// Ключ - это (AsgId, NodeId).
pub type Memo<T> = HashMap<(AsgId, NodeId), T>;

/// Трейт, определяющий общий интерфейс для исполнительной среды (бэкенда).
///
/// Любая структура, реализующая этот трейт, может взять ASG и данные
/// и выполнить вычисления, возвращая результат.
pub trait Backend {
    /// Тип, представляющий данные, специфичные для устройства (CPU или GPU).
    /// Например, для GPU это может быть обертка над wgpu::Buffer.
    type DeviceData: std::fmt::Debug;

    /// Подготавливает данные для выполнения: выделяет память на устройстве
    /// и копирует в нее данные с CPU.
    fn load_data(
        &self,
        data: &HashMap<String, Value>,
    ) -> Result<HashMap<String, Self::DeviceData>, RuntimeError>;

    /// Выполняет граф, используя и обновляя кэш вычислений.
    ///
    /// # Аргументы
    /// * `main_asg` - Основной граф для выполнения.
    /// * `initial_memo` - Кэш с начальными данными (входы, параметры) и, возможно,
    ///   результатами предыдущих вычислений (для связанных графов).
    ///
    /// # Возвращает
    /// Кортеж, содержащий:
    /// 1. Вектор с выходными данными графа.
    /// 2. Финальное состояние кэша `Memo` со всеми промежуточными результатами.
    fn run(
        &self,
        main_asg: &Asg,
        initial_memo: Memo<Self::DeviceData>,
    ) -> Result<(Vec<Self::DeviceData>, Memo<Self::DeviceData>), RuntimeError>;

    /// Забирает результат с устройства обратно в виде CPU-значения (`Value`).
    fn retrieve_data(&self, device_data: &[Self::DeviceData]) -> Result<Vec<Value>, RuntimeError>;
}