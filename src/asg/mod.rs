//! ASG: Абстрактный семантический граф

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Идентификатор узла
pub type NodeId = usize;
/// Идентификатор графа
pub type AsgId = usize;
/// Форма тензора
pub type Shape = Vec<usize>;

pub type AsgResult<T> = std::result::Result<T, AsgError>;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum AsgError {
    #[error("Узел с ID {0} не найден")]
    NodeNotFound(NodeId),
    #[error("Граф с ID {0} не найден")]
    AsgNotFound(AsgId),
    #[error("Несовместимый внешний reference")]
    InvalidExternalReference,
    #[error("Неверная форма у узла {0}")]
    InvalidShape(NodeId),
    #[error("Неверный dtype у узла {0}")]
    InvalidDType(NodeId),
    #[error("Операция не поддерживается: {0}")]
    UnsupportedOperation(String),
}

#[derive(Debug, Copy, Clone, PartialEq, Serialize, Deserialize)]
pub enum DType {
    F32,
    I32,
    Bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// ID узла (дублируем ключ HashMap для удобства вызовов в других модулях)
    pub id: NodeId,
    pub name: Option<String>,
    pub node_type: NodeType,
    pub shape: Option<Shape>,
    pub dtype: Option<DType>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeType {
    // Данные
    Input { name: String },
    Parameter { name: String },
    Literal(Value),
    External { name: String, source_asg_id: AsgId, source_node_id: NodeId },

    // Бинарные
    Add(NodeId, NodeId),
    Subtract(NodeId, NodeId),
    Multiply(NodeId, NodeId),
    Divide(NodeId, NodeId),
    // Название как ожидают другие модули:
    MatrixMultiply(NodeId, NodeId),
    GreaterThan(NodeId, NodeId),
    Less(NodeId, NodeId),
    Equal(NodeId, NodeId),
    Power(NodeId, NodeId),

    // Унарные
    Negate(NodeId),
    Exp(NodeId),
    Log(NodeId),
    Sqrt(NodeId),
    ReLU(NodeId),
    Sigmoid(NodeId),
    Softmax(NodeId),

    // Редукции
    Sum(NodeId),
    Mean(NodeId),            // по последней оси, keepdim=1
    MeanAxis(NodeId, isize), // по произвольной оси, keepdim=1
    Variance(NodeId),        // по последней оси, keepdim=1

    // Трансформации
    Reshape(NodeId, NodeId),         // второй аргумент — литерал формы
    Transpose(NodeId, usize, usize),
    Broadcast(NodeId, NodeId),
    ReduceSumTo(NodeId, NodeId),

    // Пулинг
    MaxPool2d { input: NodeId, kernel_size: (usize, usize), stride: (usize, usize) },
    // В проекте ожидается именно такая сигнатура полей:
    MaxUnpool2d { input: NodeId, original_input: NodeId, kernel_size: (usize, usize), stride: (usize, usize) },

    // Управляющие
    If { condition: NodeId, then_asg: AsgId, else_asg: AsgId },
    ForLoop { trip_count: usize, body_asg: AsgId },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Tensor(ArrayD<f32>),
    ScalarF32(f32),
    ScalarI32(i32),
    ScalarBool(bool),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Asg {
    pub id: AsgId,
    pub nodes: HashMap<NodeId, Node>,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
}

impl Asg {
    pub fn new(id: AsgId, _name: Option<String>) -> Self {
        Self { id, nodes: HashMap::new(), inputs: vec![], outputs: vec![] }
    }

    pub fn add_node(&mut self, name: Option<String>, node_type: NodeType) -> NodeId {
        let new_id = self.nodes.len();
        // Базовый узел
        let mut node = Node { id: new_id, name, node_type, shape: None, dtype: None };

        // Автопроставление shape/dtype для литералов — чтобы инференс не падал на MissingShapeInfo
        match &node.node_type {
            NodeType::Literal(Value::Tensor(arr)) => {
                node.shape = Some(arr.shape().to_vec());
                node.dtype = Some(DType::F32);
            }
            NodeType::Literal(Value::ScalarF32(_)) => {
                node.shape = Some(vec![]);
                node.dtype = Some(DType::F32);
            }
            NodeType::Literal(Value::ScalarI32(_)) => {
                node.shape = Some(vec![]);
                node.dtype = Some(DType::I32);
            }
            NodeType::Literal(Value::ScalarBool(_)) => {
                node.shape = Some(vec![]);
                node.dtype = Some(DType::Bool);
            }
            _ => {}
        }

        self.nodes.insert(new_id, node);
        new_id
    }

    pub fn set_inputs(&mut self, inputs: Vec<NodeId>) { self.inputs = inputs; }
    pub fn set_outputs(&mut self, outputs: Vec<NodeId>) { self.outputs = outputs; }

    /// Шорткат для проектов, где вызывается `set_output(node_id)`.
    pub fn set_output(&mut self, output: NodeId) {
        self.set_outputs(vec![output]);
    }

    pub fn get_node(&self, id: NodeId) -> AsgResult<&Node> {
        self.nodes.get(&id).ok_or(AsgError::NodeNotFound(id))
    }
    pub fn get_node_mut(&mut self, id: NodeId) -> AsgResult<&mut Node> {
        self.nodes.get_mut(&id).ok_or(AsgError::NodeNotFound(id))
    }
}
