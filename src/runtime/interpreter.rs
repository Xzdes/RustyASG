//! Модуль, реализующий простой интерпретатор для ASG.
//!
//! Этот интерпретатор является "эталонным" бэкендом, который выполняет
//! вычисления последовательно на CPU. Он обходит граф вычислений (ASG)
//! и для каждого узла выполняет соответствующую операцию с помощью `ndarray`.
//!
//! Основная задача интерпретатора - взять граф и конкретные входные данные
//! и вернуть конечный результат.

use crate::asg::{Asg, AsgId, NodeId, NodeType, Value};
use ndarray::{Ix2, Zip};
use std::collections::HashMap;
use thiserror::Error;

/// Ошибки, которые могут возникнуть во время выполнения (интерпретации) графа.
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
    #[error("Операция {0} еще не реализована в интерпретаторе")]
    UnimplementedOperation(String),
}

/// Контекст выполнения для одного или нескольких связанных графов.
struct ExecutionContext<'a> {
    /// Хранилище всех графов, участвующих в вычислении.
    graphs: HashMap<AsgId, &'a Asg>,
    /// Хранилище для входных данных и обучаемых параметров.
    inputs: &'a HashMap<String, Value>,
    /// Глобальный кэш для уже вычисленных значений узлов.
    /// Ключ - это (AsgId, NodeId).
    memo: HashMap<(AsgId, NodeId), Value>,
}

impl<'a> ExecutionContext<'a> {
    /// Создает новый контекст выполнения.
    fn new(main_asg: &'a Asg, inputs: &'a HashMap<String, Value>) -> Self {
        let mut graphs = HashMap::new();
        graphs.insert(main_asg.id, main_asg);
        Self {
            graphs,
            inputs,
            memo: HashMap::new(),
        }
    }

    /// Добавляет связанный граф в контекст (например, граф прямого прохода).
    pub fn add_graph(&mut self, asg: &'a Asg) {
        self.graphs.insert(asg.id, asg);
    }

    /// Главная функция, которая рекурсивно вычисляет значение для заданного узла.
    fn evaluate_node(&mut self, asg_id: AsgId, node_id: NodeId) -> Result<Value, RuntimeError> {
        // Если значение уже вычислено, возвращаем его из кэша.
        if let Some(value) = self.memo.get(&(asg_id, node_id)) {
            return Ok(value.clone());
        }

        let asg = self
            .graphs
            .get(&asg_id)
            .ok_or(RuntimeError::GraphNotFound(asg_id))?;
        let node = asg
            .nodes
            .get(&node_id)
            .ok_or(RuntimeError::NodeNotFound(node_id, asg_id))?;

        let result = match &node.node_type {
            NodeType::Input { name } => self
                .inputs
                .get(name)
                .cloned()
                .ok_or_else(|| RuntimeError::MissingInput(name.clone(), node_id)),

            NodeType::Parameter { name } => self
                .inputs
                .get(name)
                .cloned()
                .ok_or_else(|| RuntimeError::MissingParameter(name.clone(), node_id)),

            NodeType::Literal(value) => Ok(value.clone()),

            NodeType::External {
                source_asg_id,
                source_node_id,
            } => {
                // Если узел внешний, вычисляем его в контексте его собственного графа.
                // Это ключевая механика для работы графа градиентов.
                self.evaluate_node(*source_asg_id, *source_node_id)
            }

            NodeType::Add(lhs_id, rhs_id) => {
                let lhs = self.evaluate_node(asg_id, *lhs_id)?;
                let rhs = self.evaluate_node(asg_id, *rhs_id)?;
                op_add(lhs, rhs)
            }
            
            NodeType::Subtract(lhs_id, rhs_id) => {
                let lhs = self.evaluate_node(asg_id, *lhs_id)?;
                let rhs = self.evaluate_node(asg_id, *rhs_id)?;
                op_subtract(lhs, rhs)
            }

            NodeType::Multiply(lhs_id, rhs_id) => {
                let lhs = self.evaluate_node(asg_id, *lhs_id)?;
                let rhs = self.evaluate_node(asg_id, *rhs_id)?;
                op_multiply(lhs, rhs)
            }

            NodeType::MatrixMultiply(lhs_id, rhs_id) => {
                let lhs = self.evaluate_node(asg_id, *lhs_id)?;
                let rhs = self.evaluate_node(asg_id, *rhs_id)?;
                op_matmul(lhs, rhs)
            }

            NodeType::GreaterThan(lhs_id, rhs_id) => {
                let lhs = self.evaluate_node(asg_id, *lhs_id)?;
                let rhs = self.evaluate_node(asg_id, *rhs_id)?;
                op_greater_than(lhs, rhs)
            }

            NodeType::ReLU(operand_id) => {
                let operand = self.evaluate_node(asg_id, *operand_id)?;
                op_relu(operand)
            }

            NodeType::Sum(operand_id) => {
                let operand = self.evaluate_node(asg_id, *operand_id)?;
                op_sum(operand)
            }

            NodeType::Transpose(operand_id, axis1, axis2) => {
                let operand = self.evaluate_node(asg_id, *operand_id)?;
                op_transpose(operand, *axis1, *axis2)
            }

            NodeType::Power(base_id, power_id) => {
                let base = self.evaluate_node(asg_id, *base_id)?;
                let power = self.evaluate_node(asg_id, *power_id)?;
                op_power(base, power)
            }
            
            NodeType::Broadcast(source_id, target_shape_id) => {
                let source = self.evaluate_node(asg_id, *source_id)?;
                // Важно: форма для трансляции берется из узла во внешнем (исходном) графе.
                // Поэтому мы рекурсивно вызываем evaluate_node для target_shape_id,
                // который, скорее всего, будет узлом External.
                let target_shape_provider = self.evaluate_node(asg_id, *target_shape_id)?;
                op_broadcast(source, target_shape_provider)
            }

            // Заглушки для еще не реализованных операций
            _ => Err(RuntimeError::UnimplementedOperation(format!(
                "{:?}",
                node.node_type
            ))),
        }?;

        self.memo.insert((asg_id, node_id), result.clone());
        Ok(result)
    }
}

/// Публичная структура Интерпретатора.
pub struct Interpreter;

impl Interpreter {
    pub fn new() -> Self {
        Self
    }

    /// Запускает выполнение графа с заданными входами и связанными графами.
    pub fn run<'a>(
        &self,
        main_asg: &'a Asg,
        inputs: &'a HashMap<String, Value>,
        linked_graphs: &[&'a Asg],
    ) -> Result<Value, RuntimeError> {
        let mut context = ExecutionContext::new(main_asg, inputs);
        for g in linked_graphs {
            context.add_graph(g);
        }
        context.evaluate_node(main_asg.id, main_asg.output)
    }
}

impl Default for Interpreter {
    fn default() -> Self {
        Self::new()
    }
}

// --- Реализации конкретных операций ---

fn op_add(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    match (lhs, rhs) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            // ndarray поддерживает broadcasting "из коробки"
            let result = &a + &b;
            Ok(Value::Tensor(result))
        }
        _ => Err(RuntimeError::TypeError {
            expected: "Tensor and Tensor".to_string(),
            actual: "Other".to_string(),
        }),
    }
}

fn op_subtract(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    match (lhs, rhs) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            let result = &a - &b;
            Ok(Value::Tensor(result))
        }
        _ => Err(RuntimeError::TypeError {
            expected: "Tensor and Tensor".to_string(),
            actual: "Other".to_string(),
        }),
    }
}

fn op_multiply(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    match (lhs, rhs) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            let result = &a * &b;
            Ok(Value::Tensor(result))
        }
        _ => Err(RuntimeError::TypeError {
            expected: "Tensor and Tensor".to_string(),
            actual: "Other".to_string(),
        }),
    }
}

fn op_greater_than(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    match (lhs, rhs) {
        (Value::Tensor(a), Value::Tensor(b)) => {
            // Поддерживаем сравнение с другим тензором или со скаляром
            if a.shape() != b.shape() && b.ndim() != 0 {
                return Err(RuntimeError::ShapeError(format!(
                    "Incompatible shapes for GreaterThan: {:?} and {:?}",
                    a.shape(),
                    b.shape()
                )));
            }
            let mut result = a.clone();
            if b.ndim() == 0 {
                let scalar_b = b.first().unwrap();
                result.mapv_inplace(|val_a| if val_a > *scalar_b { 1.0 } else { 0.0 });
            } else {
                Zip::from(&mut result)
                    .and(&a)
                    .and(&b)
                    .for_each(|res, &val_a, &val_b| {
                        *res = if val_a > val_b { 1.0 } else { 0.0 };
                    });
            }
            Ok(Value::Tensor(result))
        }
        _ => Err(RuntimeError::TypeError {
            expected: "Tensor and Tensor".to_string(),
            actual: "Other".to_string(),
        }),
    }
}

fn op_power(base: Value, power: Value) -> Result<Value, RuntimeError> {
    match (base, power) {
        (Value::Tensor(a), Value::Tensor(b)) if b.ndim() == 0 => {
            let p = b
                .first()
                .expect("Scalar tensor for power should have one element");
            Ok(Value::Tensor(a.mapv(|val| val.powf(*p))))
        }
        _ => Err(RuntimeError::TypeError {
            expected: "Base Tensor and Scalar Tensor for power".to_string(),
            actual: "Other".to_string(),
        }),
    }
}

fn op_relu(operand: Value) -> Result<Value, RuntimeError> {
    match operand {
        Value::Tensor(a) => {
            let result = a.mapv(|val| val.max(0.0));
            Ok(Value::Tensor(result))
        }
        _ => Err(RuntimeError::TypeError {
            expected: "Tensor".to_string(),
            actual: "Other".to_string(),
        }),
    }
}

fn op_sum(operand: Value) -> Result<Value, RuntimeError> {
    use ndarray::arr0;
    match operand {
        Value::Tensor(a) => {
            let result = arr0(a.sum()).into_dyn();
            Ok(Value::Tensor(result))
        }
        _ => Err(RuntimeError::TypeError {
            expected: "Tensor".to_string(),
            actual: "Other".to_string(),
        }),
    }
}

fn op_transpose(operand: Value, axis1: usize, axis2: usize) -> Result<Value, RuntimeError> {
    match operand {
        Value::Tensor(a) => {
            let mut axes: Vec<_> = (0..a.ndim()).collect();
            if axis1 < axes.len() && axis2 < axes.len() {
                axes.swap(axis1, axis2);
                let result = a.permuted_axes(axes);
                Ok(Value::Tensor(result))
            } else {
                Err(RuntimeError::ShapeError(
                    "Invalid axes for transpose".to_string(),
                ))
            }
        }
        _ => Err(RuntimeError::TypeError {
            expected: "Tensor".to_string(),
            actual: "Other".to_string(),
        }),
    }
}

fn op_broadcast(source: Value, target_shape_provider: Value) -> Result<Value, RuntimeError> {
    let source_tensor = match source {
        Value::Tensor(val) => val,
        _ => return Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }),
    };
    let target_tensor = match target_shape_provider {
        Value::Tensor(val) => val,
        _ => return Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }),
    };

    if source_tensor.ndim() != 0 {
        return Err(RuntimeError::UnimplementedOperation("Broadcast only supports scalars for now".to_string()));
    }
    let scalar_value = *source_tensor.first().expect("Scalar tensor for broadcast source must not be empty");
    let result = ndarray::ArrayD::from_elem(target_tensor.shape(), scalar_value);
    Ok(Value::Tensor(result))
}

fn op_matmul(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    let a = match lhs {
        Value::Tensor(val) => val,
        _ => return Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }),
    };

    let b = match rhs {
        Value::Tensor(val) => val,
        _ => return Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }),
    };

    // Случай 1: Матричное умножение (dot product)
    if a.ndim() >= 2 && b.ndim() == 2 {
        let a_mat = a.view().into_dimensionality::<Ix2>()
            .map_err(|e| RuntimeError::ShapeError(format!("LHS cannot be viewed as a 2D matrix: {}", e)))?;
        let b_mat = b.view().into_dimensionality::<Ix2>()
            .map_err(|e| RuntimeError::ShapeError(format!("RHS is not a 2D matrix: {}", e)))?;

        if a_mat.shape()[1] != b_mat.shape()[0] {
            return Err(RuntimeError::ShapeError(format!(
                "Incompatible shapes for matmul: {:?} and {:?}",
                a.shape(),
                b.shape()
            )));
        }

        let result_data = a_mat.dot(&b_mat).into_dyn();
        Ok(Value::Tensor(result_data))
    }
    // Случай 2: Поэлементное умножение с трансляцией (скаляр * матрица или матрица * скаляр)
    // Это как раз тот случай, который возникает в графе градиентов.
    else if a.ndim() == 0 || b.ndim() == 0 {
        let result_data = &a * &b; // ndarray's '*' handles broadcasting.
        Ok(Value::Tensor(result_data))
    }
    // Если ни один из случаев не подошел
    else {
        Err(RuntimeError::UnimplementedOperation(format!(
            "Interpreter matmul is not implemented for dimensions {} and {}",
            a.ndim(),
            b.ndim()
        )))
    }
}