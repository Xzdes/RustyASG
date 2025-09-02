//! Модуль для автоматического дифференцирования (Autograd) на основе ASG.
//!
//! Основная идея этого модуля - преобразование графа в граф (Graph-to-Graph).
//! Мы берем ASG, который вычисляет некоторую функцию (например, потери), и
//! генерируем новый ASG, который вычисляет градиенты этой функции по
//! отношению к заданным переменным (например, весам модели).

use crate::asg::{Asg, AsgError, NodeId, NodeType, Value};
use crate::analysis::shape_inference::{ShapeInference, ShapeInferenceError};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum AutogradError {
    #[error("Ошибка графа: {0}")]
    AsgError(#[from] AsgError),
    #[error("Узел {0} не найден в графе")]
    NodeNotFound(NodeId),
    #[error("Невозможно дифференцировать операцию {0:?}")]
    NonDifferentiableOperation(NodeType),
    #[error("Ошибка вывода форм в Autograd: {0}")]
    ShapeInference(#[from] ShapeInferenceError),
}

/// Структура, отвечающая за генерацию градиента.
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
            NodeType::External {
                name,
                source_asg_id: self.source_asg.id,
                source_node_id,
            },
        );
        
        let node_mut = self.grad_asg.get_node_mut(new_node_id)?;
        node_mut.shape = source_node.shape.clone();
        node_mut.dtype = source_node.dtype;

        Ok(new_node_id)
    }

    pub fn build(
        mut self,
        target_node_id: NodeId,
        with_respect_to: &[NodeId],
    ) -> Result<Asg, AutogradError> {
        let sorted_nodes = self.topological_sort(target_node_id)?;
        
        let grad_of_target_node_id = self.grad_asg.add_node(
            None,
            NodeType::Literal(Value::Tensor(ndarray::arr0(1.0f32).into_dyn())),
        );
        self.grad_map.insert(target_node_id, grad_of_target_node_id);

        for &node_id in sorted_nodes.iter().rev() {
            if !self.grad_map.contains_key(&node_id) { continue; }

            let node = self.source_asg.get_node(node_id)?;
            let node_type = node.node_type.clone();
            let upstream_grad_id = *self.grad_map.get(&node_id).unwrap();
            
            match node_type {
                NodeType::Add(lhs_id, rhs_id) => {
                    self.accumulate_grad(lhs_id, upstream_grad_id);
                    self.accumulate_grad(rhs_id, upstream_grad_id);
                }
                NodeType::Subtract(lhs_id, rhs_id) => {
                    self.accumulate_grad(lhs_id, upstream_grad_id);
                    let neg_one = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(-1.0f32).into_dyn())));
                    let neg_grad = self.grad_asg.add_node(None, NodeType::Multiply(upstream_grad_id, neg_one));
                    self.accumulate_grad(rhs_id, neg_grad);
                }
                NodeType::Multiply(a_id, b_id) => {
                    let imported_b = self.import_node(b_id)?;
                    let grad_a = self.grad_asg.add_node(None, NodeType::Multiply(upstream_grad_id, imported_b));
                    self.accumulate_grad(a_id, grad_a);

                    let imported_a = self.import_node(a_id)?;
                    let grad_b = self.grad_asg.add_node(None, NodeType::Multiply(upstream_grad_id, imported_a));
                    self.accumulate_grad(b_id, grad_b);
                }
                NodeType::Divide(a_id, b_id) => {
                    let imported_b = self.import_node(b_id)?;
                    let grad_a = self.grad_asg.add_node(None, NodeType::Divide(upstream_grad_id, imported_b));
                    self.accumulate_grad(a_id, grad_a);
                    
                    let imported_a = self.import_node(a_id)?;
                    let neg_one = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(-1.0).into_dyn())));
                    let b_squared = self.grad_asg.add_node(None, NodeType::Multiply(imported_b, imported_b));
                    let num = self.grad_asg.add_node(None, NodeType::Multiply(upstream_grad_id, imported_a));
                    let neg_num = self.grad_asg.add_node(None, NodeType::Multiply(num, neg_one));
                    let grad_b = self.grad_asg.add_node(None, NodeType::Divide(neg_num, b_squared));
                    self.accumulate_grad(b_id, grad_b);
                }
                NodeType::Sum(x_id) => {
                    let imported_x = self.import_node(x_id)?;
                    let grad_x = self.grad_asg.add_node(None, NodeType::Broadcast(upstream_grad_id, imported_x));
                    self.accumulate_grad(x_id, grad_x);
                }
                NodeType::Sqrt(x_id) => {
                    let imported_sqrt_x = self.import_node(node_id)?;
                    let two = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(2.0).into_dyn())));
                    let den = self.grad_asg.add_node(None, NodeType::Multiply(two, imported_sqrt_x));
                    let grad_x = self.grad_asg.add_node(None, NodeType::Divide(upstream_grad_id, den));
                    self.accumulate_grad(x_id, grad_x);
                }
                NodeType::Mean(x_id) => {
                    let n_val = *self.source_asg.get_node(x_id)?.shape.as_ref().unwrap().last().unwrap_or(&1) as f32;
                    let n = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(1.0 / n_val).into_dyn())));
                    let imported_x = self.import_node(x_id)?;
                    let grad_div_n = self.grad_asg.add_node(None, NodeType::Multiply(upstream_grad_id, n));
                    let final_grad = self.grad_asg.add_node(None, NodeType::Broadcast(grad_div_n, imported_x));
                    self.accumulate_grad(x_id, final_grad);
                }
                NodeType::Variance(x_id) => {
                    let n_val = *self.source_asg.get_node(x_id)?.shape.as_ref().unwrap().last().unwrap_or(&1) as f32;
                    let n = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(1.0 / n_val).into_dyn())));
                    let imported_x = self.import_node(x_id)?;
                    let mean_x = self.grad_asg.add_node(None, NodeType::Mean(imported_x));
                    let x_minus_mean = self.grad_asg.add_node(None, NodeType::Subtract(imported_x, mean_x));
                    let two = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(2.0).into_dyn())));
                    let term1 = self.grad_asg.add_node(None, NodeType::Multiply(two, x_minus_mean));
                    let term2 = self.grad_asg.add_node(None, NodeType::Multiply(term1, n));
                    let broadcasted_grad = self.grad_asg.add_node(None, NodeType::Broadcast(upstream_grad_id, imported_x));
                    let final_grad = self.grad_asg.add_node(None, NodeType::Multiply(term2, broadcasted_grad));
                    self.accumulate_grad(x_id, final_grad);
                }
                NodeType::MaxPool2d { input, kernel_size, stride } => {
                    let original_input_imported = self.import_node(input)?;
                    let grad_x = self.grad_asg.add_node(
                        None,
                        NodeType::MaxUnpool2d {
                            input: upstream_grad_id,
                            original_input: original_input_imported,
                            kernel_size,
                            stride,
                        },
                    );
                    self.accumulate_grad(input, grad_x);
                }
                _ => { /* Пропускаем остальные */ }
            }
        }

        let mut final_outputs = Vec::new();
        for &id in with_respect_to {
            let grad_id = match self.grad_map.get(&id) {
                Some(g_id) => *g_id,
                None => self.get_or_create_zero_grad(id)?,
            };
            final_outputs.push(grad_id);
        }
        self.grad_asg.set_outputs(final_outputs);
        
        let mut initial_shapes_for_grad = HashMap::new();
        for node in self.source_asg.nodes.values() {
            if let (Some(shape), Some(dtype)) = (&node.shape, &node.dtype) {
                 let external_node_name = format!("external_{}_{}", self.source_asg.id, node.id);
                 initial_shapes_for_grad.insert(external_node_name, (shape.clone(), *dtype));
            }
        }
        ShapeInference::run(&mut self.grad_asg, &initial_shapes_for_grad)?;

        Ok(self.grad_asg)
    }

    fn accumulate_grad(&mut self, node_id: NodeId, grad_to_add_id: NodeId) {
        if let Some(&existing_grad_id) = self.grad_map.get(&node_id) {
            let new_grad_id = self.grad_asg.add_node(None, NodeType::Add(existing_grad_id, grad_to_add_id));
            self.grad_map.insert(node_id, new_grad_id);
        } else {
            self.grad_map.insert(node_id, grad_to_add_id);
        }
    }

    fn get_or_create_zero_grad(&mut self, node_id: NodeId) -> Result<NodeId, AutogradError> {
        if let Some(&grad_id) = self.grad_map.get(&node_id) {
            return Ok(grad_id);
        }

        let source_node = self.source_asg.get_node(node_id)?;
        let shape = source_node.shape.as_ref().ok_or(ShapeInferenceError::MissingInitialShape(source_node.name.clone().unwrap_or_default()))?;
        let zeros = ndarray::ArrayD::zeros(shape.clone());
        
        let zero_grad_id = self.grad_asg.add_node(
            Some(format!("zero_grad_for_{}", node_id)),
            NodeType::Literal(Value::Tensor(zeros)),
        );
        
        let node_mut = self.grad_asg.get_node_mut(zero_grad_id)?;
        node_mut.shape = Some(shape.clone());
        node_mut.dtype = source_node.dtype;

        Ok(zero_grad_id)
    }
    
    fn topological_sort(&self, start_node_id: NodeId) -> Result<Vec<NodeId>, AutogradError> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        self.build_sorted_graph(start_node_id, &mut visited, &mut sorted)?;
        Ok(sorted)
    }
    
    fn build_sorted_graph(&self, node_id: NodeId, visited: &mut HashSet<NodeId>, sorted: &mut Vec<NodeId>) -> Result<(), AutogradError> {
        if visited.contains(&node_id) { return Ok(()) }
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

        for input_id in inputs {
            self.build_sorted_graph(input_id, visited, sorted)?;
        }
        sorted.push(node_id);
        Ok(())
    }
}