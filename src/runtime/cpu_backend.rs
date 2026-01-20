//! Модуль, реализующий бэкенд для выполнения ASG на CPU.
//!
//! Этот бэкенд обходит граф вычислений (ASG) и для каждого узла
//! выполняет соответствующую операцию с помощью `ndarray`.

use super::backend::{Backend, Memo, RuntimeError};
use crate::asg::{Asg, AsgId, NodeId, NodeType, Value};
use ndarray::{s, Array4, Axis, Ix2, Zip};
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
            NodeType::Input { name } => {
                return Err(RuntimeError::MissingInput(name.clone(), node.id));
            }
            NodeType::Parameter { name } => {
                return Err(RuntimeError::MissingParameter(name.clone(), node.id));
            }
            NodeType::Literal(value) => Ok(value.clone()),
            NodeType::External { source_asg_id, source_node_id, .. } => {
                self.memo.get(&(*source_asg_id, *source_node_id))
                    .cloned()
                    .ok_or(RuntimeError::NodeNotFound(*source_node_id, *source_asg_id))
            },

            NodeType::Add(l, r) | NodeType::Subtract(l, r) | NodeType::Multiply(l, r) | NodeType::Divide(l, r) |
            NodeType::MatrixMultiply(l, r) | NodeType::GreaterThan(l, r) | NodeType::Power(l, r) |
            NodeType::Reshape(l, r) | NodeType::Broadcast(l, r) | NodeType::ReduceSumTo(l, r) => {
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
                    NodeType::ReduceSumTo(_, _) => op_reduce_sum_to(lhs, rhs),
                    _ => unreachable!(),
                }
            }

            NodeType::ReLU(op) | NodeType::Sigmoid(op) | NodeType::Softmax(op) | NodeType::Sum(op) |
            NodeType::Mean(op) | NodeType::Variance(op) | NodeType::Sqrt(op) | NodeType::Exp(op) |
            NodeType::Abs(op) | NodeType::Neg(op) | NodeType::Log(op) | NodeType::Tanh(op) |
            NodeType::GELU(op) | NodeType::SiLU(op) => {
                let operand = self.evaluate_node(asg_id, *op)?;
                match &node.node_type {
                    NodeType::ReLU(_) => op_relu(operand),
                    NodeType::Sigmoid(_) => op_sigmoid(operand),
                    NodeType::Softmax(_) => op_softmax(operand),
                    NodeType::Sum(_) => op_sum(operand),
                    NodeType::Mean(_) => op_mean(operand),
                    NodeType::Variance(_) => op_variance(operand),
                    NodeType::Sqrt(_) => op_sqrt(operand),
                    NodeType::Exp(_) => op_exp(operand),
                    NodeType::Abs(_) => op_abs(operand),
                    NodeType::Neg(_) => op_neg(operand),
                    NodeType::Log(_) => op_log(operand),
                    NodeType::Tanh(_) => op_tanh(operand),
                    NodeType::GELU(_) => op_gelu(operand),
                    NodeType::SiLU(_) => op_silu(operand),
                    _ => unreachable!(),
                }
            }

            // Операции с параметрами
            NodeType::LeakyReLU(op, slope) => {
                let operand = self.evaluate_node(asg_id, *op)?;
                op_leaky_relu(operand, *slope)
            }
            NodeType::ELU(op, alpha) => {
                let operand = self.evaluate_node(asg_id, *op)?;
                op_elu(operand, *alpha)
            }
            NodeType::Softplus(op, beta) => {
                let operand = self.evaluate_node(asg_id, *op)?;
                op_softplus(operand, *beta)
            }
            NodeType::Clamp(op, min_val, max_val) => {
                let operand = self.evaluate_node(asg_id, *op)?;
                op_clamp(operand, *min_val, *max_val)
            }
            
            NodeType::Transpose(op, ax1, ax2) => {
                let operand = self.evaluate_node(asg_id, *op)?;
                op_transpose(operand, *ax1, *ax2)
            }

            NodeType::MaxPool2d { input, kernel_size, stride } => {
                let operand = self.evaluate_node(asg_id, *input)?;
                op_max_pool2d(operand, *kernel_size, *stride)
            }

            NodeType::MaxUnpool2d { input, original_input, kernel_size, stride } => {
                let operand = self.evaluate_node(asg_id, *input)?;
                let original_operand = self.evaluate_node(asg_id, *original_input)?;
                op_max_unpool2d(operand, original_operand, *kernel_size, *stride)
            }

            NodeType::Conv2d { input, weight, bias, stride, padding, dilation, groups } => {
                let input_val = self.evaluate_node(asg_id, *input)?;
                let weight_val = self.evaluate_node(asg_id, *weight)?;
                let bias_val = if let Some(b) = bias {
                    Some(self.evaluate_node(asg_id, *b)?)
                } else {
                    None
                };
                op_conv2d(input_val, weight_val, bias_val, *stride, *padding, *dilation, *groups)
            }

            NodeType::ConvTranspose2d { input, weight, bias, stride, padding, output_padding, dilation, groups } => {
                let input_val = self.evaluate_node(asg_id, *input)?;
                let weight_val = self.evaluate_node(asg_id, *weight)?;
                let bias_val = if let Some(b) = bias {
                    Some(self.evaluate_node(asg_id, *b)?)
                } else {
                    None
                };
                op_conv_transpose2d(input_val, weight_val, bias_val, *stride, *padding, *output_padding, *dilation, *groups)
            }

            NodeType::AvgPool2d { input, kernel_size, stride, padding } => {
                let operand = self.evaluate_node(asg_id, *input)?;
                op_avg_pool2d(operand, *kernel_size, *stride, *padding)
            }

            NodeType::AvgUnpool2d { input, original_input, kernel_size, stride, padding } => {
                let grad_val = self.evaluate_node(asg_id, *input)?;
                let orig_val = self.evaluate_node(asg_id, *original_input)?;
                op_avg_unpool2d(grad_val, orig_val, *kernel_size, *stride, *padding)
            }

            NodeType::AdaptiveAvgPool2d { input, output_size } => {
                let operand = self.evaluate_node(asg_id, *input)?;
                op_adaptive_avg_pool2d(operand, *output_size)
            }

            NodeType::Embedding { indices, weight } => {
                let indices_val = self.evaluate_node(asg_id, *indices)?;
                let weight_val = self.evaluate_node(asg_id, *weight)?;
                op_embedding(indices_val, weight_val)
            }

            NodeType::EmbeddingGrad { grad_output, indices, num_embeddings } => {
                let grad_val = self.evaluate_node(asg_id, *grad_output)?;
                let indices_val = self.evaluate_node(asg_id, *indices)?;
                op_embedding_grad(grad_val, indices_val, *num_embeddings)
            }

            _ => Err(RuntimeError::UnimplementedOperation(format!("{:?}", node.node_type))),
        }?;

        self.memo.insert((asg_id, node_id), result.clone());
        Ok(result)
    }
}

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self { Self }
}

impl Default for CpuBackend {
    fn default() -> Self { Self::new() }
}

impl Backend for CpuBackend {
    type DeviceData = Value;

    fn load_data(
        &self,
        data: &HashMap<String, Value>,
    ) -> Result<HashMap<String, Self::DeviceData>, RuntimeError> {
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

        for node_id in sorted_nodes {
            context.evaluate_node(main_asg.id, node_id)?;
        }
        
        let mut results = Vec::new();
        for output_node_id in &main_asg.outputs {
            let result = context.memo.get(&(main_asg.id, *output_node_id))
                .ok_or(RuntimeError::NodeNotFound(*output_node_id, main_asg.id))?
                .clone();
            results.push(result);
        }
        Ok((results, context.memo))
    }

    fn retrieve_data(&self, device_data: &[Self::DeviceData]) -> Result<Vec<Value>, RuntimeError> {
        Ok(device_data.to_vec())
    }
}

fn op_add(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> { match (lhs, rhs) { (Value::Tensor(a), Value::Tensor(b)) => Ok(Value::Tensor(&a + &b)), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_subtract(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> { match (lhs, rhs) { (Value::Tensor(a), Value::Tensor(b)) => Ok(Value::Tensor(&a - &b)), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_multiply(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> { match (lhs, rhs) { (Value::Tensor(a), Value::Tensor(b)) => Ok(Value::Tensor(&a * &b)), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_divide(lhs: Value, rhs: Value) -> Result<Value, RuntimeError> { match (lhs, rhs) { (Value::Tensor(a), Value::Tensor(b)) => Ok(Value::Tensor(&a / &b)), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_relu(operand: Value) -> Result<Value, RuntimeError> { match operand { Value::Tensor(a) => Ok(Value::Tensor(a.mapv(|val| val.max(0.0)))), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_sigmoid(operand: Value) -> Result<Value, RuntimeError> { match operand { Value::Tensor(a) => Ok(Value::Tensor(a.mapv(|x| 1.0 / (1.0 + (-x).exp())))), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_sum(operand: Value) -> Result<Value, RuntimeError> { match operand { Value::Tensor(a) => Ok(Value::Tensor(ndarray::arr0(a.sum()).into_dyn())), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_sqrt(operand: Value) -> Result<Value, RuntimeError> { match operand { Value::Tensor(a) => Ok(Value::Tensor(a.mapv(|x| x.sqrt()))), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_exp(operand: Value) -> Result<Value, RuntimeError> { match operand { Value::Tensor(a) => Ok(Value::Tensor(a.mapv(|x| x.exp()))), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_abs(operand: Value) -> Result<Value, RuntimeError> { match operand { Value::Tensor(a) => Ok(Value::Tensor(a.mapv(|x| x.abs()))), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_neg(operand: Value) -> Result<Value, RuntimeError> { match operand { Value::Tensor(a) => Ok(Value::Tensor(a.mapv(|x| -x))), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_log(operand: Value) -> Result<Value, RuntimeError> { match operand { Value::Tensor(a) => Ok(Value::Tensor(a.mapv(|x| x.ln()))), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_tanh(operand: Value) -> Result<Value, RuntimeError> { match operand { Value::Tensor(a) => Ok(Value::Tensor(a.mapv(|x| x.tanh()))), _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }

fn op_gelu(operand: Value) -> Result<Value, RuntimeError> {
    // GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
    match operand {
        Value::Tensor(a) => Ok(Value::Tensor(a.mapv(|x| {
            0.5 * x * (1.0 + (SQRT_2_OVER_PI * (x + 0.044715 * x.powi(3))).tanh())
        }))),
        _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() })
    }
}

fn op_silu(operand: Value) -> Result<Value, RuntimeError> {
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    match operand {
        Value::Tensor(a) => Ok(Value::Tensor(a.mapv(|x| x / (1.0 + (-x).exp())))),
        _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() })
    }
}

fn op_leaky_relu(operand: Value, negative_slope: f32) -> Result<Value, RuntimeError> {
    match operand {
        Value::Tensor(a) => Ok(Value::Tensor(a.mapv(|x| if x > 0.0 { x } else { negative_slope * x }))),
        _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() })
    }
}

fn op_elu(operand: Value, alpha: f32) -> Result<Value, RuntimeError> {
    // ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
    match operand {
        Value::Tensor(a) => Ok(Value::Tensor(a.mapv(|x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) }))),
        _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() })
    }
}

fn op_softplus(operand: Value, beta: f32) -> Result<Value, RuntimeError> {
    // Softplus(x) = log(1 + exp(beta*x)) / beta
    // Для численной стабильности: if beta*x > 20, return x
    match operand {
        Value::Tensor(a) => Ok(Value::Tensor(a.mapv(|x| {
            let bx = beta * x;
            if bx > 20.0 { x } else { (1.0 + bx.exp()).ln() / beta }
        }))),
        _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() })
    }
}

fn op_clamp(operand: Value, min_val: f32, max_val: f32) -> Result<Value, RuntimeError> {
    match operand {
        Value::Tensor(a) => Ok(Value::Tensor(a.mapv(|x| x.clamp(min_val, max_val)))),
        _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() })
    }
}

fn op_mean(operand: Value) -> Result<Value, RuntimeError> { 
    match operand { 
        Value::Tensor(a) => {
            let axis = Axis(a.ndim() - 1);
            let mean = a.mean_axis(axis).unwrap();
            Ok(Value::Tensor(mean.insert_axis(axis).into_dyn()))
        }, 
        _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) 
    } 
}
fn op_variance(operand: Value) -> Result<Value, RuntimeError> { 
    match operand { 
        Value::Tensor(a) => {
            let axis = Axis(a.ndim() - 1);
            let var = a.var_axis(axis, 0.0);
            Ok(Value::Tensor(var.insert_axis(axis).into_dyn()))
        }, 
        _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) 
    } 
}
fn op_power(base: Value, power: Value) -> Result<Value, RuntimeError> { match (base, power) { (Value::Tensor(a), Value::Tensor(b)) if b.ndim() == 0 => Ok(Value::Tensor(a.mapv(|val| val.powf(*b.first().unwrap())))), _ => Err(RuntimeError::TypeError { expected: "Tensor and Scalar Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_transpose(operand: Value, axis1: usize, axis2: usize) -> Result<Value, RuntimeError> { match operand { Value::Tensor(a) => { let mut axes: Vec<_> = (0..a.ndim()).collect(); axes.swap(axis1, axis2); Ok(Value::Tensor(a.permuted_axes(axes))) }, _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }) } }
fn op_broadcast(source: Value, target: Value) -> Result<Value, RuntimeError> {
    match (source, target) {
        (Value::Tensor(s), Value::Tensor(t)) => {
            let target_shape = t.shape();
            // Если source - скаляр, заполняем target_shape этим значением
            if s.ndim() == 0 || s.len() == 1 {
                let val = *s.first().unwrap();
                Ok(Value::Tensor(ndarray::ArrayD::from_elem(target_shape, val)))
            } else {
                // Используем ndarray broadcasting
                let broadcasted = s.broadcast(target_shape)
                    .ok_or_else(|| RuntimeError::ShapeError(
                        format!("Cannot broadcast {:?} to {:?}", s.shape(), target_shape)
                    ))?;
                Ok(Value::Tensor(broadcasted.to_owned()))
            }
        },
        _ => Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() })
    }
}
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

fn op_max_pool2d(operand: Value, kernel_size: (usize, usize), stride: (usize, usize)) -> Result<Value, RuntimeError> {
    let input_tensor = match operand { Value::Tensor(val) => val, _ => return Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }), };
    let input_arr: Array4<f32> = input_tensor.into_dimensionality().map_err(|e| RuntimeError::ShapeError(e.to_string()))?;
    let (n, c, h, w) = input_arr.dim();
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let out_h = (h - kh) / sh + 1;
    let out_w = (w - kw) / sw + 1;
    let mut output_arr = Array4::<f32>::zeros((n, c, out_h, out_w));
    for n_idx in 0..n { for c_idx in 0..c { for oh_idx in 0..out_h { for ow_idx in 0..out_w {
        let h_start = oh_idx * sh;
        let w_start = ow_idx * sw;
        let window = input_arr.slice(s![n_idx, c_idx, h_start..h_start + kh, w_start..w_start + kw]);
        let max_val = window.iter().fold(f32::NEG_INFINITY, |max, &val| max.max(val));
        output_arr[[n_idx, c_idx, oh_idx, ow_idx]] = max_val;
    }}}}
    Ok(Value::Tensor(output_arr.into_dyn()))
}

fn op_max_unpool2d(operand: Value, original_input: Value, kernel_size: (usize, usize), stride: (usize, usize)) -> Result<Value, RuntimeError> {
    let grad_tensor = match operand { Value::Tensor(val) => val.into_dimensionality::<ndarray::Ix4>().map_err(|e| RuntimeError::ShapeError(e.to_string()))?, _ => return Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }), };
    let original_tensor = match original_input { Value::Tensor(val) => val.into_dimensionality::<ndarray::Ix4>().map_err(|e| RuntimeError::ShapeError(e.to_string()))?, _ => return Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }), };
    let (_n, _c, h, w) = original_tensor.dim();
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let mut output_arr = Array4::<f32>::zeros((_n, _c, h, w));
    for n_idx in 0..grad_tensor.dim().0 { for c_idx in 0..grad_tensor.dim().1 { for oh_idx in 0..grad_tensor.dim().2 { for ow_idx in 0..grad_tensor.dim().3 {
        let h_start = oh_idx * sh;
        let w_start = ow_idx * sw;
        let window = original_tensor.slice(s![n_idx, c_idx, h_start..h_start + kh, w_start..w_start + kw]);
        let grad_val = grad_tensor[[n_idx, c_idx, oh_idx, ow_idx]];
        let mut max_val = f32::NEG_INFINITY;
        let mut max_pos = (0, 0);
        for r in 0..kh { for col in 0..kw { if window[[r, col]] > max_val { max_val = window[[r, col]]; max_pos = (r, col); } } }
        output_arr[[n_idx, c_idx, h_start + max_pos.0, w_start + max_pos.1]] += grad_val;
    }}}}
    Ok(Value::Tensor(output_arr.into_dyn()))
}

fn op_reduce_sum_to(source: Value, target_shape_provider: Value) -> Result<Value, RuntimeError> {
    let mut source_tensor = match source {
        Value::Tensor(val) => val,
        _ => return Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }),
    };
    let target_shape = match target_shape_provider {
        Value::Tensor(val) => val.shape().to_vec(),
        _ => return Err(RuntimeError::TypeError { expected: "Tensor".to_string(), actual: "Other".to_string() }),
    };

    let source_rank = source_tensor.ndim();
    let target_rank = target_shape.len();
    
    if source_rank > target_rank {
        let rank_diff = source_rank - target_rank;
        for _ in 0..rank_diff {
            source_tensor = source_tensor.sum_axis(Axis(0));
        }
    }
    
    let mut axes_to_sum = Vec::new();
    let current_shape = source_tensor.shape();
    let rank_diff = current_shape.len() - target_rank;
    for i in 0..target_rank {
        if target_shape[i] == 1 && current_shape[i + rank_diff] > 1 {
            axes_to_sum.push(i + rank_diff);
        }
    }
    
    for &axis in axes_to_sum.iter().rev() {
        source_tensor = source_tensor.sum_axis(Axis(axis));
    }
    
    // ИСПРАВЛЕНО: Заменяем .into_shape() на .to_shape()?.to_owned()
    source_tensor
        .to_shape(target_shape)
        .map_err(|e| RuntimeError::ShapeError(e.to_string()))
        .map(|view| Value::Tensor(view.to_owned()))
}

/// 2D Convolution implementation using im2col approach.
/// Input: [N, C_in, H, W], Weight: [C_out, C_in/groups, kH, kW], Bias: [C_out]
fn op_conv2d(
    input: Value,
    weight: Value,
    bias: Option<Value>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> Result<Value, RuntimeError> {
    let input_arr: Array4<f32> = match input {
        Value::Tensor(val) => val.into_dimensionality()
            .map_err(|e| RuntimeError::ShapeError(format!("Conv2d input: {}", e)))?,
        _ => return Err(RuntimeError::TypeError {
            expected: "Tensor".to_string(),
            actual: "Other".to_string()
        }),
    };

    let weight_arr: Array4<f32> = match weight {
        Value::Tensor(val) => val.into_dimensionality()
            .map_err(|e| RuntimeError::ShapeError(format!("Conv2d weight: {}", e)))?,
        _ => return Err(RuntimeError::TypeError {
            expected: "Tensor".to_string(),
            actual: "Other".to_string()
        }),
    };

    let (batch_size, in_channels, in_h, in_w) = input_arr.dim();
    let (out_channels, weight_in_channels, kernel_h, kernel_w) = weight_arr.dim();

    // Validate dimensions
    if in_channels != weight_in_channels * groups {
        return Err(RuntimeError::ShapeError(format!(
            "Conv2d: input channels {} != weight_in_channels {} * groups {}",
            in_channels, weight_in_channels, groups
        )));
    }

    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;
    let (dil_h, dil_w) = dilation;

    // Calculate output dimensions
    let effective_kernel_h = (kernel_h - 1) * dil_h + 1;
    let effective_kernel_w = (kernel_w - 1) * dil_w + 1;
    let out_h = (in_h + 2 * pad_h - effective_kernel_h) / stride_h + 1;
    let out_w = (in_w + 2 * pad_w - effective_kernel_w) / stride_w + 1;

    let mut output = Array4::<f32>::zeros((batch_size, out_channels, out_h, out_w));

    let out_channels_per_group = out_channels / groups;
    let in_channels_per_group = in_channels / groups;

    for n in 0..batch_size {
        for g in 0..groups {
            let in_ch_start = g * in_channels_per_group;
            let out_ch_start = g * out_channels_per_group;

            for oc in 0..out_channels_per_group {
                let out_ch = out_ch_start + oc;
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;

                        for ic in 0..weight_in_channels {
                            let in_ch = in_ch_start + ic;
                            for kh in 0..kernel_h {
                                for kw in 0..kernel_w {
                                    let ih = (oh * stride_h + kh * dil_h) as isize - pad_h as isize;
                                    let iw = (ow * stride_w + kw * dil_w) as isize - pad_w as isize;

                                    if ih >= 0 && ih < in_h as isize && iw >= 0 && iw < in_w as isize {
                                        sum += input_arr[[n, in_ch, ih as usize, iw as usize]]
                                             * weight_arr[[out_ch, ic, kh, kw]];
                                    }
                                }
                            }
                        }

                        output[[n, out_ch, oh, ow]] = sum;
                    }
                }
            }
        }
    }

    // Add bias if present
    if let Some(bias_val) = bias {
        let bias_arr = match bias_val {
            Value::Tensor(val) => val,
            _ => return Err(RuntimeError::TypeError {
                expected: "Tensor".to_string(),
                actual: "Other".to_string()
            }),
        };

        for n in 0..batch_size {
            for c in 0..out_channels {
                let b = bias_arr[c];
                for h in 0..out_h {
                    for w in 0..out_w {
                        output[[n, c, h, w]] += b;
                    }
                }
            }
        }
    }

    Ok(Value::Tensor(output.into_dyn()))
}

/// Transposed 2D Convolution (Deconvolution).
fn op_conv_transpose2d(
    input: Value,
    weight: Value,
    bias: Option<Value>,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
    dilation: (usize, usize),
    groups: usize,
) -> Result<Value, RuntimeError> {
    let input_arr: Array4<f32> = match input {
        Value::Tensor(val) => val.into_dimensionality()
            .map_err(|e| RuntimeError::ShapeError(format!("ConvTranspose2d input: {}", e)))?,
        _ => return Err(RuntimeError::TypeError {
            expected: "Tensor".to_string(),
            actual: "Other".to_string()
        }),
    };

    let weight_arr: Array4<f32> = match weight {
        Value::Tensor(val) => val.into_dimensionality()
            .map_err(|e| RuntimeError::ShapeError(format!("ConvTranspose2d weight: {}", e)))?,
        _ => return Err(RuntimeError::TypeError {
            expected: "Tensor".to_string(),
            actual: "Other".to_string()
        }),
    };

    let (batch_size, in_channels, in_h, in_w) = input_arr.dim();
    let (weight_out_channels, out_channels_per_group, kernel_h, kernel_w) = weight_arr.dim();

    let (stride_h, stride_w) = stride;
    let (pad_h, pad_w) = padding;
    let (out_pad_h, out_pad_w) = output_padding;
    let (dil_h, dil_w) = dilation;

    let out_channels = out_channels_per_group * groups;

    // Calculate output dimensions for transposed conv
    let out_h = (in_h - 1) * stride_h - 2 * pad_h + dil_h * (kernel_h - 1) + out_pad_h + 1;
    let out_w = (in_w - 1) * stride_w - 2 * pad_w + dil_w * (kernel_w - 1) + out_pad_w + 1;

    let mut output = Array4::<f32>::zeros((batch_size, out_channels, out_h, out_w));

    let in_channels_per_group = in_channels / groups;

    for n in 0..batch_size {
        for g in 0..groups {
            let in_ch_start = g * in_channels_per_group;
            let out_ch_start = g * out_channels_per_group;

            for ic_rel in 0..in_channels_per_group {
                let ic = in_ch_start + ic_rel;
                for ih in 0..in_h {
                    for iw in 0..in_w {
                        let in_val = input_arr[[n, ic, ih, iw]];

                        for oc_rel in 0..out_channels_per_group {
                            let oc = out_ch_start + oc_rel;
                            for kh in 0..kernel_h {
                                for kw in 0..kernel_w {
                                    let oh = ih * stride_h + kh * dil_h;
                                    let ow = iw * stride_w + kw * dil_w;

                                    if oh >= pad_h && ow >= pad_w {
                                        let out_oh = oh - pad_h;
                                        let out_ow = ow - pad_w;
                                        if out_oh < out_h && out_ow < out_w {
                                            output[[n, oc, out_oh, out_ow]] +=
                                                in_val * weight_arr[[ic_rel + g * in_channels_per_group, oc_rel, kh, kw]];
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Add bias if present
    if let Some(bias_val) = bias {
        let bias_arr = match bias_val {
            Value::Tensor(val) => val,
            _ => return Err(RuntimeError::TypeError {
                expected: "Tensor".to_string(),
                actual: "Other".to_string()
            }),
        };

        for n in 0..batch_size {
            for c in 0..out_channels {
                let b = bias_arr[c];
                for h in 0..out_h {
                    for w in 0..out_w {
                        output[[n, c, h, w]] += b;
                    }
                }
            }
        }
    }

    Ok(Value::Tensor(output.into_dyn()))
}

/// Average Pooling 2D.
fn op_avg_pool2d(
    operand: Value,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Value, RuntimeError> {
    let input_arr: Array4<f32> = match operand {
        Value::Tensor(val) => val.into_dimensionality()
            .map_err(|e| RuntimeError::ShapeError(e.to_string()))?,
        _ => return Err(RuntimeError::TypeError {
            expected: "Tensor".to_string(),
            actual: "Other".to_string()
        }),
    };

    let (n, c, h, w) = input_arr.dim();
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (pad_h, pad_w) = padding;

    let out_h = (h + 2 * pad_h - kh) / sh + 1;
    let out_w = (w + 2 * pad_w - kw) / sw + 1;

    let mut output = Array4::<f32>::zeros((n, c, out_h, out_w));
    let kernel_area = (kh * kw) as f32;

    for n_idx in 0..n {
        for c_idx in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut sum = 0.0f32;
                    let mut count = 0;

                    for khh in 0..kh {
                        for kww in 0..kw {
                            let ih = (oh * sh + khh) as isize - pad_h as isize;
                            let iw = (ow * sw + kww) as isize - pad_w as isize;

                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                sum += input_arr[[n_idx, c_idx, ih as usize, iw as usize]];
                                count += 1;
                            }
                        }
                    }

                    // Use actual count for edge cases with padding
                    output[[n_idx, c_idx, oh, ow]] = if count > 0 {
                        sum / count as f32
                    } else {
                        0.0
                    };
                }
            }
        }
    }

    Ok(Value::Tensor(output.into_dyn()))
}

/// Adaptive Average Pooling 2D.
/// Automatically calculates kernel size and stride to achieve target output size.
fn op_adaptive_avg_pool2d(
    operand: Value,
    output_size: (usize, usize),
) -> Result<Value, RuntimeError> {
    let input_arr: Array4<f32> = match operand {
        Value::Tensor(val) => val.into_dimensionality()
            .map_err(|e| RuntimeError::ShapeError(e.to_string()))?,
        _ => return Err(RuntimeError::TypeError {
            expected: "Tensor".to_string(),
            actual: "Other".to_string()
        }),
    };

    let (n, c, in_h, in_w) = input_arr.dim();
    let (out_h, out_w) = output_size;

    let mut output = Array4::<f32>::zeros((n, c, out_h, out_w));

    for n_idx in 0..n {
        for c_idx in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    // Calculate input region for this output pixel
                    let h_start = (oh * in_h) / out_h;
                    let h_end = ((oh + 1) * in_h) / out_h;
                    let w_start = (ow * in_w) / out_w;
                    let w_end = ((ow + 1) * in_w) / out_w;

                    let mut sum = 0.0f32;
                    let count = (h_end - h_start) * (w_end - w_start);

                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            sum += input_arr[[n_idx, c_idx, ih, iw]];
                        }
                    }

                    output[[n_idx, c_idx, oh, ow]] = sum / count as f32;
                }
            }
        }
    }

    Ok(Value::Tensor(output.into_dyn()))
}

/// Embedding lookup: converts indices to dense vectors.
/// Indices: any shape [*], Weight: [num_embeddings, embedding_dim]
/// Output: [*, embedding_dim]
fn op_embedding(indices: Value, weight: Value) -> Result<Value, RuntimeError> {
    let indices_arr = match indices {
        Value::Tensor(val) => val,
        _ => return Err(RuntimeError::TypeError {
            expected: "Tensor".to_string(),
            actual: "Other".to_string()
        }),
    };

    let weight_arr = match weight {
        Value::Tensor(val) => val,
        _ => return Err(RuntimeError::TypeError {
            expected: "Tensor".to_string(),
            actual: "Other".to_string()
        }),
    };

    // Weight должен быть 2D: [num_embeddings, embedding_dim]
    if weight_arr.ndim() != 2 {
        return Err(RuntimeError::ShapeError(format!(
            "Embedding weight must be 2D, got {}D", weight_arr.ndim()
        )));
    }

    let num_embeddings = weight_arr.shape()[0];
    let embedding_dim = weight_arr.shape()[1];

    // Вычисляем форму выхода: indices.shape + [embedding_dim]
    let indices_shape = indices_arr.shape();
    let mut output_shape: Vec<usize> = indices_shape.to_vec();
    output_shape.push(embedding_dim);

    // Создаем выходной тензор
    let total_indices = indices_arr.len();
    let mut output_data = vec![0.0f32; total_indices * embedding_dim];

    // Извлекаем embedding'и для каждого индекса
    for (i, &idx_f32) in indices_arr.iter().enumerate() {
        let idx = idx_f32 as usize;
        if idx >= num_embeddings {
            return Err(RuntimeError::ShapeError(format!(
                "Embedding index {} out of bounds for num_embeddings {}", idx, num_embeddings
            )));
        }

        // Копируем вектор embedding'а в выходной массив
        for j in 0..embedding_dim {
            output_data[i * embedding_dim + j] = weight_arr[[idx, j]];
        }
    }

    let output = ndarray::ArrayD::from_shape_vec(
        ndarray::IxDyn(&output_shape),
        output_data
    ).map_err(|e| RuntimeError::ShapeError(e.to_string()))?;

    Ok(Value::Tensor(output))
}

/// Backward pass for AvgPool2d.
/// Distributes gradient uniformly across the pooling window.
fn op_avg_unpool2d(
    grad: Value,
    original_input: Value,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Value, RuntimeError> {
    let grad_arr: Array4<f32> = match grad {
        Value::Tensor(val) => val.into_dimensionality()
            .map_err(|e| RuntimeError::ShapeError(e.to_string()))?,
        _ => return Err(RuntimeError::TypeError {
            expected: "Tensor".to_string(),
            actual: "Other".to_string()
        }),
    };

    let orig_arr: Array4<f32> = match original_input {
        Value::Tensor(val) => val.into_dimensionality()
            .map_err(|e| RuntimeError::ShapeError(e.to_string()))?,
        _ => return Err(RuntimeError::TypeError {
            expected: "Tensor".to_string(),
            actual: "Other".to_string()
        }),
    };

    let (n, c, in_h, in_w) = orig_arr.dim();
    let (grad_n, grad_c, out_h, out_w) = grad_arr.dim();
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;
    let (ph, pw) = padding;

    // Output has same shape as original input
    let mut output = Array4::<f32>::zeros((n, c, in_h, in_w));

    let window_size = (kh * kw) as f32;

    for n_idx in 0..n {
        for c_idx in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let grad_val = grad_arr[[n_idx, c_idx, oh, ow]];
                    let distributed_grad = grad_val / window_size;

                    // Distribute to each position in the window
                    for kh_idx in 0..kh {
                        for kw_idx in 0..kw {
                            let ih = oh * sh + kh_idx;
                            let iw = ow * sw + kw_idx;

                            // Account for padding
                            if ih >= ph && ih < in_h + ph && iw >= pw && iw < in_w + pw {
                                let actual_ih = ih - ph;
                                let actual_iw = iw - pw;
                                if actual_ih < in_h && actual_iw < in_w {
                                    output[[n_idx, c_idx, actual_ih, actual_iw]] += distributed_grad;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(Value::Tensor(output.into_dyn()))
}

/// Embedding gradient: scatter-add operation.
/// Accumulates gradients into the weight matrix based on indices.
/// grad_output: [*, embedding_dim], indices: [*]
/// Output: [num_embeddings, embedding_dim]
fn op_embedding_grad(
    grad_output: Value,
    indices: Value,
    num_embeddings: usize,
) -> Result<Value, RuntimeError> {
    let grad_arr = match grad_output {
        Value::Tensor(val) => val,
        _ => return Err(RuntimeError::TypeError {
            expected: "Tensor".to_string(),
            actual: "Other".to_string()
        }),
    };

    let indices_arr = match indices {
        Value::Tensor(val) => val,
        _ => return Err(RuntimeError::TypeError {
            expected: "Tensor".to_string(),
            actual: "Other".to_string()
        }),
    };

    // grad_output shape: [*, embedding_dim]
    // indices shape: [*]
    // Последняя размерность grad_output - это embedding_dim
    let grad_shape = grad_arr.shape();
    let embedding_dim = *grad_shape.last().unwrap();

    // Создаем выходной тензор [num_embeddings, embedding_dim]
    let mut output = ndarray::ArrayD::zeros(ndarray::IxDyn(&[num_embeddings, embedding_dim]));

    // Scatter-add: для каждого индекса добавляем соответствующий градиент
    let total_indices = indices_arr.len();

    for i in 0..total_indices {
        let idx = indices_arr.as_slice().unwrap()[i] as usize;
        if idx >= num_embeddings {
            return Err(RuntimeError::ShapeError(format!(
                "EmbeddingGrad: index {} out of bounds for num_embeddings {}", idx, num_embeddings
            )));
        }

        // Добавляем градиент для этого индекса
        for j in 0..embedding_dim {
            output[[idx, j]] += grad_arr.as_slice().unwrap()[i * embedding_dim + j];
        }
    }

    Ok(Value::Tensor(output))
}