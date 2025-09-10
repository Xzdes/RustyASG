// --- Файл: src/autograd/mod.rs ---

//! Модуль для автоматического дифференцирования (Autograd) на основе ASG.

use crate::asg::{Asg, AsgError, NodeId, NodeType, Shape, Value};
use crate::analysis::shape_inference::{ShapeInference, ShapeInferenceError};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum AutogradError {
    #[error("Ошибка графа: {0}")]
    AsgError(#[from] AsgError),
    #[error("Ошибка вывода форм в Autograd: {0}")]
    ShapeInference(#[from] ShapeInferenceError),
}

pub struct Gradients {
    source_asg: Asg,
    grad_asg: Asg,
    grad_map: HashMap<NodeId, NodeId>,
}

impl Gradients {
    pub fn new(source_asg: Asg) -> Self {
        let grad_asg_id = source_asg.id + 1;
        Self {
            source_asg,
            grad_asg: Asg::new(grad_asg_id, Some("grad_graph".to_string())),
            grad_map: HashMap::new(),
        }
    }

    fn import_node(&mut self, source_node_id: NodeId) -> Result<NodeId, AutogradError> {
        let source_node = self.source_asg.get_node(source_node_id)?;
        let name = format!("external_{}_{}", self.source_asg.id, source_node_id);
        
        let new_node_id = self.grad_asg.add_node(
            Some(name.clone()),
            NodeType::External { name, source_asg_id: self.source_asg.id, source_node_id },
        );
        
        let node_mut = self.grad_asg.get_node_mut(new_node_id)?;
        node_mut.shape = source_node.shape.clone();
        node_mut.dtype = source_node.dtype;

        Ok(new_node_id)
    }

    /// **ФИНАЛЬНАЯ ИСПРАВЛЕННАЯ ВЕРСИЯ**
    /// Сворачивает градиент до нужной формы, если имело место расширение (broadcasting).
    fn reduce_grad_for_broadcast(
        &mut self,
        grad_id: NodeId,
        grad_shape: &Shape,
        target_shape: &Shape,
    ) -> Result<NodeId, AutogradError> {
        if grad_shape == target_shape {
            return Ok(grad_id);
        }

        let mut current_grad_id = grad_id;
        let mut current_shape = grad_shape.clone();

        // Шаг 1: Суммируем по осям, которые были добавлены спереди (например, [B, D] -> [D])
        let rank_diff = current_shape.len().saturating_sub(target_shape.len());
        if rank_diff > 0 {
            for _ in 0..rank_diff {
                let n = current_shape[0] as f32;
                let rank = current_shape.len();
                let transposed_id = self.grad_asg.add_node(None, NodeType::Transpose(current_grad_id, 0, rank - 1));
                let mean_id = self.grad_asg.add_node(None, NodeType::Mean(transposed_id));
                let n_const_id = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(n).into_dyn())));
                
                current_grad_id = self.grad_asg.add_node(None, NodeType::Multiply(mean_id, n_const_id));
                current_shape.remove(0); // Ранг уменьшился на 1
            }
        }
        
        // Шаг 2: Суммируем по осям, которые были расширены с 1 (например, [B, D] -> [B, 1])
        let mut axes_to_sum = Vec::new();
        // Важно: target_shape может быть короче, используем zip
        for (i, (current_dim, target_dim)) in current_shape.iter().zip(target_shape.iter()).enumerate() {
            if *target_dim == 1 && *current_dim > 1 {
                axes_to_sum.push(i);
            }
        }
        
        // Итерируем в обратном порядке, чтобы индексы осей не сдвигались после редукции
        for axis in axes_to_sum.into_iter().rev() {
            let n = current_shape[axis] as f32;
            let rank = current_shape.len();
            let transposed_id = self.grad_asg.add_node(None, NodeType::Transpose(current_grad_id, axis, rank - 1));
            let mean_id = self.grad_asg.add_node(None, NodeType::Mean(transposed_id));
            let n_const_id = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(n).into_dyn())));
            
            // Сумма = Mean * N. Результат будет иметь ранг-1.
            current_grad_id = self.grad_asg.add_node(None, NodeType::Multiply(mean_id, n_const_id));
            
            // Восстанавливаем ранг, добавляя ось размером 1
            let mut new_shape_vec = current_shape.clone();
            new_shape_vec.remove(axis);
            new_shape_vec.insert(axis, 1);
            
            // ИСПРАВЛЕНИЕ: Используем ndarray::Array::from(vec).into_dyn()
            let shape_vec: Vec<f32> = new_shape_vec.iter().map(|&d| d as f32).collect();
            let shape_tensor = Value::Tensor(ndarray::Array::from(shape_vec).into_dyn());
            
            let shape_node_id = self.grad_asg.add_node(None, NodeType::Literal(shape_tensor));
            current_grad_id = self.grad_asg.add_node(None, NodeType::Reshape(current_grad_id, shape_node_id));
            
            current_shape = new_shape_vec;
        }

        Ok(current_grad_id)
    }

    pub fn build(mut self, target_node_id: NodeId, with_respect_to: &[NodeId]) -> Result<Asg, AutogradError> {
        let sorted_nodes = self.topological_sort(target_node_id)?;
        let grad_of_target_node_id = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(1.0f32).into_dyn())));
        self.grad_map.insert(target_node_id, grad_of_target_node_id);

        for &node_id in sorted_nodes.iter().rev() {
            if !self.grad_map.contains_key(&node_id) { continue; }
            let node = self.source_asg.get_node(node_id)?;
            let upstream_grad_id = *self.grad_map.get(&node_id).unwrap();
            
            match node.node_type.clone() {
                NodeType::Add(lhs_id, rhs_id) | NodeType::Subtract(lhs_id, rhs_id) => {
                    let upstream_shape = self.source_asg.get_node(node_id)?.shape.as_ref().unwrap().clone();
                    let lhs_shape = self.source_asg.get_node(lhs_id)?.shape.as_ref().unwrap().clone();
                    let rhs_shape = self.source_asg.get_node(rhs_id)?.shape.as_ref().unwrap().clone();
                    let is_subtract = matches!(node.node_type, NodeType::Subtract(_, _));

                    let grad_lhs = self.reduce_grad_for_broadcast(upstream_grad_id, &upstream_shape, &lhs_shape)?;
                    self.accumulate_grad(lhs_id, grad_lhs)?;

                    let grad_rhs_raw = if is_subtract {
                        let neg_one = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(-1.0).into_dyn())));
                        self.grad_asg.add_node(None, NodeType::Multiply(upstream_grad_id, neg_one))
                    } else { upstream_grad_id };
                    let grad_rhs = self.reduce_grad_for_broadcast(grad_rhs_raw, &upstream_shape, &rhs_shape)?;
                    self.accumulate_grad(rhs_id, grad_rhs)?;
                },
                NodeType::Multiply(a_id, b_id) => {
                    let upstream_shape = self.source_asg.get_node(node_id)?.shape.as_ref().unwrap().clone();
                    let a_shape = self.source_asg.get_node(a_id)?.shape.as_ref().unwrap().clone();
                    let imported_b = self.import_node(b_id)?;
                    let grad_a_raw = self.grad_asg.add_node(None, NodeType::Multiply(upstream_grad_id, imported_b));
                    let grad_a = self.reduce_grad_for_broadcast(grad_a_raw, &upstream_shape, &a_shape)?;
                    self.accumulate_grad(a_id, grad_a)?;

                    let b_shape = self.source_asg.get_node(b_id)?.shape.as_ref().unwrap().clone();
                    let imported_a = self.import_node(a_id)?;
                    let grad_b_raw = self.grad_asg.add_node(None, NodeType::Multiply(upstream_grad_id, imported_a));
                    let grad_b = self.reduce_grad_for_broadcast(grad_b_raw, &upstream_shape, &b_shape)?;
                    self.accumulate_grad(b_id, grad_b)?;
                },
                NodeType::Divide(a_id, b_id) => {
                    let upstream_shape = self.source_asg.get_node(node_id)?.shape.as_ref().unwrap().clone();
                    let a_shape = self.source_asg.get_node(a_id)?.shape.as_ref().unwrap().clone();
                    let imported_b = self.import_node(b_id)?;
                    let grad_a_raw = self.grad_asg.add_node(None, NodeType::Divide(upstream_grad_id, imported_b));
                    let grad_a = self.reduce_grad_for_broadcast(grad_a_raw, &upstream_shape, &a_shape)?;
                    self.accumulate_grad(a_id, grad_a)?;
                    
                    let b_shape = self.source_asg.get_node(b_id)?.shape.as_ref().unwrap().clone();
                    let imported_a = self.import_node(a_id)?;
                    let neg_one = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(-1.0).into_dyn())));
                    let b_squared = self.grad_asg.add_node(None, NodeType::Multiply(imported_b, imported_b));
                    let num = self.grad_asg.add_node(None, NodeType::Multiply(upstream_grad_id, imported_a));
                    let neg_num = self.grad_asg.add_node(None, NodeType::Multiply(num, neg_one));
                    let grad_b_raw = self.grad_asg.add_node(None, NodeType::Divide(neg_num, b_squared));
                    let grad_b = self.reduce_grad_for_broadcast(grad_b_raw, &upstream_shape, &b_shape)?;
                    self.accumulate_grad(b_id, grad_b)?;
                },
                NodeType::Sum(x_id) => {
                    let imported_x = self.import_node(x_id)?;
                    let grad_x = self.grad_asg.add_node(None, NodeType::Broadcast(upstream_grad_id, imported_x));
                    self.accumulate_grad(x_id, grad_x)?;
                },
                NodeType::Sqrt(x_id) => {
                    let imported_sqrt_x = self.import_node(node_id)?;
                    let two = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(2.0).into_dyn())));
                    let den = self.grad_asg.add_node(None, NodeType::Multiply(two, imported_sqrt_x));
                    let grad_x = self.grad_asg.add_node(None, NodeType::Divide(upstream_grad_id, den));
                    self.accumulate_grad(x_id, grad_x)?;
                },
                NodeType::Mean(x_id) => {
                    let x_shape = self.source_asg.get_node(x_id)?.shape.as_ref().unwrap();
                    let n_val = *x_shape.last().unwrap_or(&1) as f32;
                    if n_val == 0.0 { continue; }
                    let n_const = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(1.0 / n_val).into_dyn())));
                    let imported_x = self.import_node(x_id)?;
                    let broadcasted_upstream = self.grad_asg.add_node(None, NodeType::Broadcast(upstream_grad_id, imported_x));
                    let final_grad = self.grad_asg.add_node(None, NodeType::Multiply(broadcasted_upstream, n_const));
                    self.accumulate_grad(x_id, final_grad)?;
                },
                NodeType::Variance(x_id) => {
                    let x_shape = self.source_asg.get_node(x_id)?.shape.as_ref().unwrap();
                    let n_val = *x_shape.last().unwrap_or(&1) as f32;
                    if n_val == 0.0 { continue; }
                    let imported_x = self.import_node(x_id)?;
                    let mean_node = self.grad_asg.add_node(None, NodeType::Mean(imported_x));
                    let x_minus_mean = self.grad_asg.add_node(None, NodeType::Subtract(imported_x, mean_node));
                    let two_div_n_const = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(2.0 / n_val).into_dyn())));
                    let local_grad = self.grad_asg.add_node(None, NodeType::Multiply(two_div_n_const, x_minus_mean));
                    let broadcasted_upstream = self.grad_asg.add_node(None, NodeType::Broadcast(upstream_grad_id, imported_x));
                    let final_grad = self.grad_asg.add_node(None, NodeType::Multiply(broadcasted_upstream, local_grad));
                    self.accumulate_grad(x_id, final_grad)?;
                },
                NodeType::MaxPool2d { input, kernel_size, stride } => {
                    let original_input_imported = self.import_node(input)?;
                    let grad_x = self.grad_asg.add_node(
                        None, NodeType::MaxUnpool2d { input: upstream_grad_id, original_input: original_input_imported, kernel_size, stride, },
                    );
                    self.accumulate_grad(input, grad_x)?;
                },
                NodeType::ReLU(x_id) => {
                    let imported_x = self.import_node(x_id)?;
                    let zero = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(0.0).into_dyn())));
                    let condition = self.grad_asg.add_node(None, NodeType::GreaterThan(imported_x, zero));
                    let grad_x = self.grad_asg.add_node(None, NodeType::Multiply(upstream_grad_id, condition));
                    self.accumulate_grad(x_id, grad_x)?;
                },
                _ => {}
            }
        }

        let mut final_outputs = Vec::new();
        for &id in with_respect_to {
            final_outputs.push(self.get_or_create_zero_grad(id)?);
        }
        self.grad_asg.set_outputs(final_outputs);
        
        let mut initial_shapes_for_grad = HashMap::new();
        for node in self.source_asg.nodes.values() {
            if let (Some(shape), Some(dtype)) = (&node.shape, &node.dtype) {
                 initial_shapes_for_grad.insert(format!("external_{}_{}", self.source_asg.id, node.id), (shape.clone(), *dtype));
            }
        }
        ShapeInference::run(&mut self.grad_asg, &initial_shapes_for_grad)?;

        Ok(self.grad_asg)
    }

    fn accumulate_grad(&mut self, node_id: NodeId, grad_to_add: NodeId) -> Result<(), AutogradError> {
        if let Some(&existing_grad) = self.grad_map.get(&node_id) {
            let new_grad = self.grad_asg.add_node(None, NodeType::Add(existing_grad, grad_to_add));
            self.grad_map.insert(node_id, new_grad);
        } else {
            self.grad_map.insert(node_id, grad_to_add);
        }
        Ok(())
    }

    fn get_or_create_zero_grad(&mut self, node_id: NodeId) -> Result<NodeId, AutogradError> {
        if let Some(&grad_id) = self.grad_map.get(&node_id) { return Ok(grad_id); }
        let source_node = self.source_asg.get_node(node_id)?;
        let shape = source_node.shape.as_ref().ok_or(ShapeInferenceError::MissingInitialShape(source_node.name.clone().unwrap_or_default()))?;
        let zeros = ndarray::ArrayD::zeros(shape.clone());
        let zero_grad_id = self.grad_asg.add_node(Some(format!("zero_grad_for_{}", node_id)), NodeType::Literal(Value::Tensor(zeros)));
        let node_mut = self.grad_asg.get_node_mut(zero_grad_id)?;
        node_mut.shape = Some(shape.clone());
        node_mut.dtype = source_node.dtype;
        Ok(zero_grad_id)
    }
    
    fn topological_sort(&self, start_node_id: NodeId) -> Result<Vec<NodeId>, AutogradError> {
        let mut sorted = Vec::new(); let mut visited = HashSet::new();
        self.build_sorted_graph(start_node_id, &mut visited, &mut sorted)?;
        Ok(sorted)
    }
    
    fn build_sorted_graph(&self, node_id: NodeId, visited: &mut HashSet<NodeId>, sorted: &mut Vec<NodeId>) -> Result<(), AutogradError> {
        if visited.contains(&node_id) { return Ok(()); }
        if !self.source_asg.nodes.contains_key(&node_id) { return Ok(()); }
        visited.insert(node_id);
        let node = self.source_asg.get_node(node_id)?;
        let inputs = match &node.node_type {
            NodeType::Add(a, b) | NodeType::Subtract(a, b) | NodeType::Multiply(a, b) | NodeType::Divide(a, b) |
            NodeType::MatrixMultiply(a, b) | NodeType::GreaterThan(a, b) | NodeType::Power(a, b) |
            NodeType::Broadcast(a, b) | NodeType::Reshape(a, b) => vec![*a, *b],
            NodeType::ReLU(a) | NodeType::Sum(a) | NodeType::Sigmoid(a) | NodeType::Softmax(a) |
            NodeType::Mean(a) | NodeType::Variance(a) | NodeType::Sqrt(a) | NodeType::Log(a) => vec![*a],
            NodeType::Transpose(a, _, _) => vec![*a],
            NodeType::MaxPool2d { input, .. } => vec![*input],
            _ => vec![],
        };
        for &input_id in &inputs { self.build_sorted_graph(input_id, visited, sorted)?; }
        sorted.push(node_id);
        Ok(())
    }
}