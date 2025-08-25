//! Модуль, определяющий ядро Абстрактного Семантического Графа (ASG).
//!
//! ASG - это фундаментальное представление любой вычислительной задачи в
//! экосистеме `RustyGradients`. Вместо немедленного выполнения операций,
//! фреймворк сначала строит граф, который описывает данные и последовательность
//! операций над ними.
//!
//! Этот подход позволяет проводить сложный анализ графа, его оптимизацию,
//! JIT-компиляцию для различных бэкендов (CPU, GPU) и автоматическое
//! дифференцирование.
//!
//! # Ключевые компоненты:
//!
//! - `Asg`: Сам граф, коллекция узлов и информация о его входах/выходах.
//! - `Node`: Узел в графе, представляющий либо данные, либо операцию.
//! - `NodeType`: Перечисление всех возможных операций (математика, логика, I/O).
//! - `Value`: Перечисление всех возможных типов данных, с которыми оперирует граф.
//! - `NodeId`, `AsgId`: Уникальные идентификаторы для узлов и графов.

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Уникальный идентификатор для узла в графе.
pub type NodeId = usize;

/// Уникальный идентификатор для самого графа (полезно для вложенных графов).
pub type AsgId = usize;

/// Тип `Result` для операций, связанных с ASG.
pub type AsgResult<T> = std::result::Result<T, AsgError>;

/// Ошибки, которые могут возникнуть при работе с ASG.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum AsgError {
    #[error("Узел с ID {0} не найден в графе")]
    NodeNotFound(NodeId),
    #[error("Вход с именем '{0}' не найден в графе")]
    InputNotFound(String),
}

/// Перечисление всех возможных типов данных (значений), которые могут
/// существовать в графе.
///
/// ASG спроектирован для работы с мультимодальными данными, а не только с тензорами.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// Стандартный многомерный массив (тензор) для численных вычислений.
    Tensor(ArrayD<f32>),
    /// 64-битное целое число.
    Integer(i64),
    /// 32-битное число с плавающей запятой.
    Float(f32),
    /// Логическое значение.
    Boolean(bool),
    /// Строка текста.
    Text(String),
    /// Тип "пустота", аналог `()` в Rust. Используется для операций, которые
    /// не возвращают значимого результата (например, `Print`).
    Unit,
}

/// Узел в Абстрактном Семантическом Графе.
///
/// Каждый узел имеет уникальный ID и тип, который определяет, является ли узел
/// данными (например, `Literal`) или операцией (`MatrixMultiply`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// Уникальный ID узла внутри его графа.
    pub id: NodeId,
    /// Необязательное имя для отладки и визуализации.
    pub name: Option<String>,
    /// Тип узла, определяющий его поведение.
    pub node_type: NodeType,
}

/// Перечисление всех возможных операций и типов данных в графе.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeType {
    // --- Узлы данных и входов ---
    /// Входной узел графа. Определяет публичный API графа.
    Input { name: String },
    /// Обучаемый параметр модели (например, веса или смещения).
    Parameter { name: String },
    /// Константное значение (литерал), встроенное прямо в граф.
    Literal(Value),
    /// Ссылка на узел в другом, "внешнем" графе.
    /// Используется в графе градиентов, чтобы ссылаться на значения
    /// из графа прямого прохода.
    External {
        source_asg_id: AsgId,
        source_node_id: NodeId,
    },

    // --- Математические и логические операции ---
    Add(NodeId, NodeId),
    Subtract(NodeId, NodeId),
    Multiply(NodeId, NodeId),
    MatrixMultiply(NodeId, NodeId),
    GreaterThan(NodeId, NodeId), // Поэлементное сравнение >

    // --- Поэлементные операции ---
    ReLU(NodeId),
    Sigmoid(NodeId),
    Log(NodeId),
    Power(NodeId, NodeId), // Второй аргумент - степень (может быть константой)

    // --- Операции свертки/редукции ---
    Sum(NodeId), // Сумма всех элементов тензора

    // --- Операции трансформации ---
    Reshape(NodeId, NodeId), // Второй аргумент - тензор с новой формой
    Transpose(NodeId, usize, usize), // Оси для транспонирования

    // --- Управляющие конструкции ---
    /// Условное выполнение. Выполняет один из двух под-графов в зависимости
    /// от условия.
    If {
        condition: NodeId,
        then_asg: AsgId,
        else_asg: AsgId,
    },
    /// Циклическое выполнение под-графа.
    ForLoop {
        iterable: NodeId, // Узел, по которому итерируемся (например, тензор)
        loop_body_asg: AsgId, // ID под-графа, который будет телом цикла
    },

    // --- Функции ---
    /// Определение функции внутри графа.
    FunctionDefinition {
        name: String,
        body_asg: AsgId,
    },
    /// Вызов ранее определенной функции.
    FunctionCall {
        function_id: NodeId, // ID узла FunctionDefinition
        args: Vec<NodeId>,
    },

    // --- Ввод/Вывод и побочные эффекты ---
    /// Печатает значение узла в стандартный вывод во время выполнения.
    Print(NodeId),
}

/// Структура, представляющая сам Абстрактный Семантический Граф.
///
/// Содержит коллекцию всех узлов и определяет "интерфейс" графа —
/// его входы и выходной узел.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Asg {
    /// Уникальный ID графа.
    pub id: AsgId,
    /// Необязательное имя для отладки.
    pub name: Option<String>,
    /// Все узлы, принадлежащие этому графу, хранящиеся по их ID.
    pub nodes: HashMap<NodeId, Node>,
    /// ID узлов, которые являются входами этого графа.
    pub inputs: Vec<NodeId>,
    /// ID узла, который является результатом вычисления всего графа.
    pub output: NodeId,
}

impl Asg {
    /// Создает новый, пустой граф с заданным ID.
    pub fn new(id: AsgId, name: Option<String>) -> Self {
        Self {
            id,
            name,
            nodes: HashMap::new(),
            inputs: Vec::new(),
            // Инициализируем "невалидным" значением, которое нужно будет установить.
            output: NodeId::MAX,
        }
    }

    /// Добавляет новый узел в граф.
    pub fn add_node(&mut self, name: Option<String>, node_type: NodeType) -> NodeId {
        // Простой инкрементальный ID для узлов
        let new_id = self.nodes.len();
        let node = Node {
            id: new_id,
            name,
            node_type,
        };
        self.nodes.insert(new_id, node);
        new_id
    }

    /// Устанавливает входные узлы для графа.
    pub fn set_inputs(&mut self, inputs: Vec<NodeId>) {
        self.inputs = inputs;
    }

    /// Устанавливает выходной узел для графа.
    pub fn set_output(&mut self, output: NodeId) {
        self.output = output;
    }

    /// Находит узел по его ID.
    pub fn get_node(&self, id: NodeId) -> AsgResult<&Node> {
        self.nodes
            .get(&id)
            .ok_or(AsgError::NodeNotFound(id))
    }
}