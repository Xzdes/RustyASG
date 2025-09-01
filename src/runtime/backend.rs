//! Модуль, определяющий абстрактный интерфейс (трейт) для исполнительных бэкендов.

use crate::asg::{Asg, AsgId, NodeId, Value};
use std::collections::HashMap;
use thiserror::Error;

/// Ошибки, которые могут возникнуть во время выполнения (интерпретации) графа.
/// Этот тип ошибок является общим для всех бэкендов.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum RuntimeError {
    #[error("Узел с ID {0} (в графе {1}) не найден")]
    NodeNotFound(NodeId, AsgId),
    #[error("Граф с ID {0} не найден в контексте выполнения")]
    GraphNotFound(AsgId),
    #[error("Неверный тип значения для операции: ожидался {expected}, получен {actual}")]
    TypeError { expected: String, actual: String },
    #[error("Несовместимые формы тензоров для операции: {0}")]
    ShapeError(String),
    #[error("Для выполнения графа не предоставлено значение для входа '{0}' (ID: {1})")]
    MissingInput(String, NodeId),
    #[error("Для выполнения графа не предоставлено значение для параметра '{0}' (ID: {1})")]
    MissingParameter(String, NodeId),
    #[error("Операция {0} еще не реализована в этом бэкенде")]
    UnimplementedOperation(String),
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