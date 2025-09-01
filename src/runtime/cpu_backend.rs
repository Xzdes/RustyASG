//! Модуль, реализующий бэкенд для выполнения ASG на CPU.
//!
//! Этот бэкенд обходит граф вычислений (ASG) и для каждого узла
//! выполняет соответствующую операцию с помощью `ndarray`.

use super::backend::{Backend, Memo, RuntimeError};
use crate::asg::{Asg, AsgId, NodeId, NodeType, Value};
use ndarray::{s, Axis, Ix2, Zip};
use std::collections::HashMap;

/// Контекст выполнения для одного или нескольких связанных графов на CPU.
struct ExecutionContext<'a> {
    /// Хранилище всех графов, участвующих в вычислении.
    graphs: HashMap<AsgId, &'a Asg>,
    /// Глобальный кэш для уже вычисленных значений узлов.
    /// Ключ - это (AsgId, NodeId).
    memo: Memo<Value>,
}

impl<'a> ExecutionContext<'a> {
    /// Создает новый контекст выполнения.
    fn new(main_asg: &'a Asg, initial_memo: Memo<Value>) -> Self {
        let mut graphs = HashMap::new();
        graphs.insert(main_asg.id, main_asg);
        Self {
            graphs,
            memo: initial_memo,
        }
    }

    /// Главная функция, которая рекурсивно вычисляет значение для заданного узла.
    fn evaluate_node(&mut self, asg_id: AsgId, node_id: NodeId) -> Result<Value, RuntimeError> {
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
            NodeType::Input { .. } | NodeType::Parameter { .. } => {
                 // Входы и параметры должны быть уже в memo к моменту вызова `run`
                 return Err(RuntimeError::MissingInput(
                    node.name.clone().unwrap_or_default(),
                    node_id
                ));
            }
            NodeType::Literal(value) => Ok(value.clone()),
            NodeType::External { source_asg_id, source_node_id, .. } => {
                // Пытаемся получить значение из кэша. Если его там нет, это ошибка.
                self.memo.get(&(*source_asg_id, *source_node_id))
                    .cloned()
                    .ok_or(RuntimeError::NodeNotFound(*source_node_id, *source_asg_id))
            },

            // Бинарные операции
            NodeType::Add(l, r) | NodeType::Subtract(l, r) | NodeType::Multiply(l, r) | NodeType::Divide(l, r) |
            NodeType::MatrixMultiply(l, r) | NodeType::GreaterThan(l, r) | NodeType::Power(l, r) |
            NodeType::Reshape(l, r) | NodeType::Broadcast(l, r) => {
                let lhs = self.evaluate_node(asg_id, *l)?;
                let rhs = self.evaluate_node(asg_id, *r)?;
                match &node.node_type {
                    NodeType::Add(_, _) => op_add(lhs, rhs),
                    NodeType::Subtract(_, _) => op_subtract(lhs, rhs),
                    NodeType::Multiply(_, _) => op_multiply(lhs, rhs),
                    NodeType::Divide(_, _) => op_divide(lhs, rhs),
                    NodeType::MatrixMultiply(_, _) => op_matmul(lhs, rhs),
                    NodeType::GreaterThan(_, _) => op_greater_than(lhs, rhs),
                    NodeType::Power(_, _) => op_power(lhs, rhs),
                    NodeType::Reshape(_, _) => op_reshape(lhs, rhs),
                    NodeType::Broadcast(_, _) => op_broadcast(lhs, rhs),
                    _ => unreachable!(),
                }
            }

            // Унарные операции
            NodeType::ReLU(op) | NodeType::Sigmoid(op) | NodeType::Softmax(op) | NodeType::Sum(op) |
            NodeType::Mean(op) | NodeType::Variance(op) | NodeType::Sqrt(op) => {
                let operand = self.evaluate_node(asg_id, *op)?;
                match &node.node_type {
                    NodeType::ReLU(_) => op_relu(operand),
                    NodeType::Sigmoid(_) => op_sigmoid(operand),
                    NodeType::Softmax(_) => op_softmax(operand),
                    NodeType::Sum(_) => op_sum(operand),
                    NodeType::Mean(_) => op_mean(operand),
                    NodeType::Variance(_) => op_variance(operand),
                    NodeType::Sqrt(_) => op_sqrt(operand),
                    _ => unreachable!(),
                }
            }
            
            NodeType::Transpose(op, ax1, ax2) => {
                let operand = self.evaluate_node(asg_id, *op)?;
                op_transpose(operand, *ax1, *ax2)
            }

            _ => Err(RuntimeError::UnimplementedOperation(format!("{:?}", node.node_type))),
        }?;

        self.memo.insert((asg_id, node_id), result.clone());
        Ok(result)
    }
}

/// Исполнительный бэкенд, работающий на CPU.
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self { Self }
}

impl Default for CpuBackend {
    fn default() -> Self { Self::new() }
}

impl Backend for CpuBackend {
    /// Для CPU-бэкенда "данные на устройстве" - это обычные `Value`.
    type DeviceData = Value;

    fn load_data(
        &self,
        data: &HashMap<String, Value>,
    ) -> Result<HashMap<String, Self::DeviceData>, RuntimeError> {
        // Копирование на CPU не требуется, просто клонируем HashMap.
        Ok(data.clone())
    }

    fn run(
        &self,
        main_asg: &Asg,
        initial_memo: Memo<Self::DeviceData>,
    ) -> Result<(Vec<Self::DeviceData>, Memo<Self::DeviceData>), RuntimeError> {
        let sorted_nodes = crate::analysis::shape_inference::ShapeInference::topological_sort(main_asg)
            .map_err(|e| RuntimeError::ShapeError(format!("Topological sort failed: {:?}", e)))?;

        let mut context = ExecutionContext::new(main_asg, initial_memo);

        // Последовательно вычисляем каждый узел
        for node_id in sorted_nodes {
            context.evaluate_node(main_asg.id, node_id)?;
        }
        
        // Собираем результаты
        let mut results = Vec::new();
        for output_node_id in &main_asg.outputs {
            let result = context.memo.get(&(main_asg.id, *output_node_id))
                .ok_or(RuntimeError::NodeNotFound(*output_node_id, main_asg.id))?
                .clone();
            results.push(result);
        }
        // Возвращаем результаты и итоговый кэш
        Ok((results, context.memo))
    }

    fn retrieve_data(&self, device_data: &[Self::DeviceData]) -> Result<Vec<Value>, RuntimeError> {
        // Загрузка с CPU не требуется, просто клонируем данные.
        Ok(device_data.to_vec())
    }
}


// --- Реализации операций ---

fn op_add(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> { match (lhs, rhs) { (Value::Tensor(a), Value::Tensor(b)) => Ok(Value::Tensor(&a + &b)), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_subtract(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> { match (lhs, rhs) { (Value::Tensor(a), Value::Tensor(b)) => Ok(Value::Tensor(&a - &b)), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_multiply(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> { match (lhs, rhs) { (Value::Tensor(a), Value::Tensor(b)) => Ok(Value::Tensor(&a * &b)), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_divide(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> { match (lhs, rhs) { (Value::Tensor(a), Value::Tensor(b)) => Ok(Value::Tensor(&a / &b)), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_relu(operand: Value) -> Result<Value, RuntimeError> { match operand { Value::Tensor(a) => Ok(Value::Tensor(a.mapv(|val| val.max(0.0)))), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_sigmoid(operand: Value) -> Result<Value, RuntimeError> { match operand { Value::Tensor(a) => Ok(Value::Tensor(a.mapv(|x| 1.0 / (1.0 + (-x).exp())))), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_sum(operand: Value) -> Result<Value, RuntimeError> { match operand { Value::Tensor(a) => Ok(Value::Tensor(ndarray::arr0(a.sum()).into_dyn())), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_sqrt(operand: Value) -> Result<Value, RuntimeError> { match operand { Value::Tensor(a) => Ok(Value::Tensor(a.mapv(|x| x.sqrt()))), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_mean(operand: Value) -> Result<Value, RuntimeError> { match operand { Value::Tensor(a) => Ok(Value::Tensor(a.mean_axis(Axis(a.ndim() - 1)).unwrap().into_dyn())), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_variance(operand: Value) -> Result<Value, RuntimeError> { match operand { Value::Tensor(a) => Ok(Value::Tensor(a.var_axis(Axis(a.ndim() - 1), 0.0).into_dyn())), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_power(base: Value, power: Value) -> Result<Value, RuntimeError> { match (base, power) { (Value::Tensor(a), Value::Tensor(b)) if b.ndim() == 0 => Ok(Value::Tensor(a.mapv(|val| val.powf(*b.first().unwrap())))), _ => Err(RuntimeError::TypeError { expected: "Tensor and Scalar Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_transpose(operand: Value, axis1: usize, axis2: usize) -> Result<Value, RuntimeError> { match operand { Value::Tensor(a) => { let mut axes: Vec<_> = (0..a.ndim()).collect(); axes.swap(axis1, axis2); Ok(Value::Tensor(a.permuted_axes(axes))) }, _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_broadcast(source: Value, target: Value) -> Result<Value, RuntimeError> { match (source, target) { (Value::Tensor(s), Value::Tensor(t)) => Ok(Value::Tensor(ndarray::ArrayD::from_elem(t.shape(), *s.first().unwrap()))), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_reshape(source: Value, shape_provider: Value) -> Result<Value, RuntimeError> { match (source, shape_provider) { (Value::Tensor(s), Value::Tensor(p)) => { let shape: Vec<usize> = p.iter().map(|&x| x as usize).collect(); let reshaped = s.to_shape(shape.as_slice()).map_err(|e| RuntimeError::ShapeError(e.to_string()))?; Ok(Value::Tensor(reshaped.to_owned())) }, _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_greater_than(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> { match (lhs, rhs) { (Value::Tensor(a), Value::Tensor(b)) => { let mut r = a.clone(); if b.ndim() == 0 { r.mapv_inplace(|v| if v > *b.first().unwrap() { 1.0 } else { 0.0 }); } else { Zip::from(&mut r).and(&a).and(&b).for_each(|res, &va, &vb| *res = if va > vb { 1.0 } else { 0.0 }); } Ok(Value::Tensor(r)) }, _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_softmax(operand: Value) -> Result<Value, RuntimeError> { match operand { Value::Tensor(a) => { let mut result = a.clone(); let last_axis = Axis(a.ndim() - 1); result.axis_iter_mut(last_axis).for_each(|mut row| { let max_val = row.iter().fold(f32::NEG_INFINITY, |max, &val| max.max(val)); row.mapv_inplace(|x| (x - max_val).exp()); let sum = row.sum(); row.mapv_inplace(|x| x / sum); }); Ok(Value::Tensor(result)) }, _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_matmul(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> {
    let a = match lhs { Value::Tensor(val) => val, _ => return Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) };
    let b = match rhs { Value::Tensor(val) => val, _ => return Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) };
    if a.ndim() == 4 && b.ndim() == 4 {
        let (b0,b1,m,_,n) = (a.shape()[0],a.shape()[1],a.shape()[2],a.shape()[3],b.shape()[3]);
        let mut out = ndarray::ArrayD::zeros(ndarray::IxDyn(&[b0,b1,m,n]));
        for i in 0..b0 { for j in 0..b1 {
            let a_mat=a.slice(s![i,j,..,..]).into_dimensionality::<Ix2>().unwrap();
            let b_mat=b.slice(s![i,j,..,..]).into_dimensionality::<Ix2>().unwrap();
            out.slice_mut(s![i,j,..,..]).assign(&a_mat.dot(&b_mat));
        }}
        return Ok(Value::Tensor(out));
    } else if a.ndim() >= 2 && b.ndim() == 2 {
        let a_mat=a.view().into_dimensionality::<Ix2>().unwrap();let b_mat=b.view().into_dimensionality::<Ix2>().unwrap();
        if a_mat.shape()[1]!=b_mat.shape()[0]{return Err(RuntimeError::ShapeError(format!("Incompatible matmul shapes: {:?} and {:?}", a.shape(), b.shape())));}
        return Ok(Value::Tensor(a_mat.dot(&b_mat).into_dyn()));
    } else if a.ndim() == 0 || b.ndim() == 0 { return Ok(Value::Tensor(&a * &b)); }
    Err(RuntimeError::UnimplementedOperation(format!("Matmul for dims {} and {}", a.ndim(), b.ndim())))
}