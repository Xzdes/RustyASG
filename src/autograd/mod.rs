//! Модуль для автоматического дифференцирования (Autograd) на основе ASG.
//!
//! Основная идея этого модуля - преобразование графа в граф (Graph-to-Graph).
//! Мы берем ASG, который вычисляет некоторую функцию (например, потери), и
//! генерируем новый ASG, который вычисляет градиенты этой функции по
//! отношению к заданным переменным (например, весам модели).

use crate::asg::{Asg, AsgError, NodeId, NodeType, Value};
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
    #[error("Shape-информация отсутствует в исходном графе для узла {0}, что необходимо для Autograd")]
    MissingShapeInfo(NodeId),
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

    fn import_node(&mut self, source_node_id: NodeId) -> NodeId {
        let source_node = self.source_asg.get_node(source_node_id).unwrap();
        let name = format!("external_{}_{}", self.source_asg.id, source_node_id);
        
        let new_node_id = self.grad_asg.add_node(
            Some(name.clone()),
            NodeType::External {
                name,
                source_asg_id: self.source_asg.id,
                source_node_id,
            },
        );
        
        let node_mut = self.grad_asg.get_node_mut(new_node_id).unwrap();
        node_mut.shape = source_node.shape.clone();
        node_mut.dtype = source_node.dtype;

        new_node_id
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
            if !self.grad_map.contains_key(&node_id) {
                continue;
            }

            let node = self.source_asg.get_node(node_id)?;
            let node_type = node.node_type.clone();
            let grad_shape = node.shape.as_ref().ok_or(AutogradError::MissingShapeInfo(node_id))?.clone();
            
            let upstream_grad_id = self.get_or_create_zero_grad(node_id);

            match node_type {
                NodeType::Add(lhs_id, rhs_id) => {
                    let lhs_shape = self.source_asg.get_node(lhs_id)?.shape.as_ref().ok_or(AutogradError::MissingShapeInfo(lhs_id))?.clone();
                    let mut grad_lhs = upstream_grad_id;
                    if lhs_shape != grad_shape {
                        grad_lhs = self.grad_asg.add_node(Some("sum_grad_for_lhs".to_string()), NodeType::Sum(grad_lhs));
                    }
                    self.accumulate_grad(lhs_id, grad_lhs);

                    let rhs_shape = self.source_asg.get_node(rhs_id)?.shape.as_ref().ok_or(AutogradError::MissingShapeInfo(rhs_id))?.clone();
                    let mut grad_rhs = upstream_grad_id;
                    if rhs_shape != grad_shape {
                        grad_rhs = self.grad_asg.add_node(Some("sum_grad_for_rhs".to_string()), NodeType::Sum(grad_rhs));
                    }
                    self.accumulate_grad(rhs_id, grad_rhs);
                }
                NodeType::Subtract(lhs_id, rhs_id) => {
                    let lhs_shape = self.source_asg.get_node(lhs_id)?.shape.as_ref().ok_or(AutogradError::MissingShapeInfo(lhs_id))?.clone();
                    let mut grad_lhs = upstream_grad_id;
                    if lhs_shape != grad_shape {
                        grad_lhs = self.grad_asg.add_node(Some("sum_grad_for_lhs".to_string()), NodeType::Sum(grad_lhs));
                    }
                    self.accumulate_grad(lhs_id, grad_lhs);

                    let neg_one = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(-1.0f32).into_dyn())));
                    let neg_upstream_grad = self.grad_asg.add_node(None, NodeType::Multiply(upstream_grad_id, neg_one));

                    let rhs_shape = self.source_asg.get_node(rhs_id)?.shape.as_ref().ok_or(AutogradError::MissingShapeInfo(rhs_id))?.clone();
                    let mut grad_rhs = neg_upstream_grad;
                    if rhs_shape != grad_shape {
                       grad_rhs = self.grad_asg.add_node(Some("sum_grad_for_rhs".to_string()), NodeType::Sum(grad_rhs));
                    }
                    self.accumulate_grad(rhs_id, grad_rhs);
                }
                NodeType::Multiply(a_id, b_id) => {
                    let a_shape = self.source_asg.get_node(a_id)?.shape.as_ref().ok_or(AutogradError::MissingShapeInfo(a_id))?.clone();
                    let imported_b = self.import_node(b_id);
                    let mut grad_a = self.grad_asg.add_node(None, NodeType::Multiply(upstream_grad_id, imported_b));
                    if a_shape != grad_shape {
                        grad_a = self.grad_asg.add_node(Some("sum_grad_for_a".to_string()), NodeType::Sum(grad_a));
                    }
                    self.accumulate_grad(a_id, grad_a);
                    
                    let b_shape = self.source_asg.get_node(b_id)?.shape.as_ref().ok_or(AutogradError::MissingShapeInfo(b_id))?.clone();
                    let imported_a = self.import_node(a_id);
                    let mut grad_b = self.grad_asg.add_node(None, NodeType::Multiply(upstream_grad_id, imported_a));
                    if b_shape != grad_shape {
                        grad_b = self.grad_asg.add_node(Some("sum_grad_for_b".to_string()), NodeType::Sum(grad_b));
                    }
                    self.accumulate_grad(b_id, grad_b);
                }
                NodeType::Divide(a_id, b_id) => {
                    let imported_b = self.import_node(b_id);
                    let grad_a_full = self.grad_asg.add_node(None, NodeType::Divide(upstream_grad_id, imported_b));
                    let a_shape = self.source_asg.get_node(a_id)?.shape.as_ref().ok_or(AutogradError::MissingShapeInfo(a_id))?.clone();
                    let mut grad_a = grad_a_full;
                    if a_shape != grad_shape {
                        grad_a = self.grad_asg.add_node(Some("sum_grad_for_a".to_string()), NodeType::Sum(grad_a));
                    }
                    self.accumulate_grad(a_id, grad_a);

                    let imported_a = self.import_node(a_id);
                    let neg_one = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(-1.0f32).into_dyn())));
                    let b_squared = self.grad_asg.add_node(None, NodeType::Multiply(imported_b, imported_b));
                    let term1 = self.grad_asg.add_node(None, NodeType::Multiply(neg_one, upstream_grad_id));
                    let term2 = self.grad_asg.add_node(None, NodeType::Multiply(term1, imported_a));
                    let grad_b_full = self.grad_asg.add_node(None, NodeType::Divide(term2, b_squared));
                    let b_shape = self.source_asg.get_node(b_id)?.shape.as_ref().ok_or(AutogradError::MissingShapeInfo(b_id))?.clone();
                    let mut grad_b = grad_b_full;
                    if b_shape != grad_shape {
                        grad_b = self.grad_asg.add_node(Some("sum_grad_for_b".to_string()), NodeType::Sum(grad_b));
                    }
                    self.accumulate_grad(b_id, grad_b);
                }
                NodeType::Power(base_id, power_id) => {
                    let imported_base = self.import_node(base_id);
                    let imported_power = self.import_node(power_id);
                    let one = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(1.0f32).into_dyn())));
                    let power_minus_one = self.grad_asg.add_node(None, NodeType::Subtract(imported_power, one));
                    let base_to_power = self.grad_asg.add_node(None, NodeType::Power(imported_base, power_minus_one));
                    let term1 = self.grad_asg.add_node(None, NodeType::Multiply(imported_power, base_to_power));
                    let local_grad = self.grad_asg.add_node(None, NodeType::Multiply(upstream_grad_id, term1));
                    self.accumulate_grad(base_id, local_grad);
                }
                NodeType::MatrixMultiply(a_id, b_id) => {
                    let (a_rank, b_rank) = {
                        let a_node = self.source_asg.get_node(a_id)?;
                        let b_node = self.source_asg.get_node(b_id)?;
                        (
                            a_node.shape.as_ref().ok_or(AutogradError::MissingShapeInfo(a_id))?.len(),
                            b_node.shape.as_ref().ok_or(AutogradError::MissingShapeInfo(b_id))?.len()
                        )
                    };
                    
                    let imported_a = self.import_node(a_id);
                    let imported_b = self.import_node(b_id);

                    let b_transposed = if b_rank >= 2 {
                        self.grad_asg.add_node(None, NodeType::Transpose(imported_b, b_rank - 2, b_rank - 1))
                    } else { imported_b };
                    let grad_a = self.grad_asg.add_node(None, NodeType::MatrixMultiply(upstream_grad_id, b_transposed));
                    self.accumulate_grad(a_id, grad_a);

                    let a_transposed = if a_rank >= 2 {
                        self.grad_asg.add_node(None, NodeType::Transpose(imported_a, a_rank - 2, a_rank - 1))
                    } else { imported_a };
                    let grad_b = self.grad_asg.add_node(None, NodeType::MatrixMultiply(a_transposed, upstream_grad_id));
                    self.accumulate_grad(b_id, grad_b);
                }
                NodeType::Sum(x_id) => {
                    let imported_x = self.import_node(x_id);
                    let grad_x = self.grad_asg.add_node(None, NodeType::Broadcast(upstream_grad_id, imported_x));
                    self.accumulate_grad(x_id, grad_x);
                }
                NodeType::Mean(x_id) => {
                    let n_val = {
                        let x_node = self.source_asg.get_node(x_id)?;
                        let x_shape = x_node.shape.as_ref().ok_or(AutogradError::MissingShapeInfo(x_id))?;
                        *x_shape.last().unwrap_or(&1) as f32
                    };
                    
                    let n = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(n_val).into_dyn())));
                    let grad_div_n = self.grad_asg.add_node(None, NodeType::Divide(upstream_grad_id, n));
                    
                    let imported_x = self.import_node(x_id);
                    let final_grad = self.grad_asg.add_node(None, NodeType::Broadcast(grad_div_n, imported_x));
                    self.accumulate_grad(x_id, final_grad);
                }
                NodeType::Variance(x_id) => {
                    let n_val = {
                        let x_node = self.source_asg.get_node(x_id)?;
                        let x_shape = x_node.shape.as_ref().ok_or(AutogradError::MissingShapeInfo(x_id))?;
                        *x_shape.last().unwrap_or(&1) as f32
                    };

                    let n = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(n_val).into_dyn())));
                    let imported_x = self.import_node(x_id);
                    
                    let mean_node = self.grad_asg.add_node(None, NodeType::Mean(imported_x));
                    let x_minus_mean = self.grad_asg.add_node(None, NodeType::Subtract(imported_x, mean_node));
                    
                    let two = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(2.0).into_dyn())));
                    let two_mul_diff = self.grad_asg.add_node(None, NodeType::Multiply(two, x_minus_mean));
                    
                    let local_grad_unscaled = self.grad_asg.add_node(None, NodeType::Divide(two_mul_diff, n));
                    
                    let broadcasted_upstream_grad = self.grad_asg.add_node(None, NodeType::Broadcast(upstream_grad_id, imported_x));
                    let final_grad = self.grad_asg.add_node(None, NodeType::Multiply(local_grad_unscaled, broadcasted_upstream_grad));

                    self.accumulate_grad(x_id, final_grad);
                }
                NodeType::ReLU(x_id) => {
                    let imported_x = self.import_node(x_id);
                    let zero = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(0.0f32).into_dyn())));
                    let step_mask_id = self.grad_asg.add_node(None, NodeType::GreaterThan(imported_x, zero));
                    let local_grad_id = self.grad_asg.add_node(None, NodeType::Multiply(upstream_grad_id, step_mask_id));
                    self.accumulate_grad(x_id, local_grad_id);
                }
                NodeType::Softmax(x_id) => {
                    let s = self.import_node(node_id);
                    let grad_s = upstream_grad_id;

                    let mul_term = self.grad_asg.add_node(None, NodeType::Multiply(grad_s, s));
                    let sum_term = self.grad_asg.add_node(None, NodeType::Sum(mul_term));
                    let broadcast_sum = self.grad_asg.add_node(None, NodeType::Broadcast(sum_term, s));
                    let sub_term = self.grad_asg.add_node(None, NodeType::Subtract(grad_s, broadcast_sum));
                    let grad_x = self.grad_asg.add_node(None, NodeType::Multiply(s, sub_term));

                    self.accumulate_grad(x_id, grad_x);
                }
                NodeType::Transpose(x_id, axis1, axis2) => {
                    let grad_x = self.grad_asg.add_node(None, NodeType::Transpose(upstream_grad_id, axis1, axis2));
                    self.accumulate_grad(x_id, grad_x);
                }
                NodeType::Reshape(x_id, _shape_node_id) => {
                    let x_shape = self.source_asg.get_node(x_id)?
                        .shape.as_ref()
                        .ok_or(AutogradError::MissingShapeInfo(x_id))?
                        .clone();
                    
                    let shape_data: Vec<f32> = x_shape.iter().map(|&dim| dim as f32).collect();
                    let shape_tensor = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[x_shape.len()]), shape_data)
                        .expect("Failed to create shape tensor");
                    let new_shape_node_id = self.grad_asg.add_node(
                        None, 
                        NodeType::Literal(Value::Tensor(shape_tensor))
                    );

                    let grad_x = self.grad_asg.add_node(None, NodeType::Reshape(upstream_grad_id, new_shape_node_id));
                    self.accumulate_grad(x_id, grad_x);
                }
                NodeType::Sqrt(x_id) => {
                    let sqrt_x = self.import_node(node_id);
                    let two = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(2.0f32).into_dyn())));
                    let denominator = self.grad_asg.add_node(None, NodeType::Multiply(two, sqrt_x));
                    let local_grad = self.grad_asg.add_node(None, NodeType::Divide(upstream_grad_id, denominator));
                    self.accumulate_grad(x_id, local_grad);
                }
                NodeType::Input { .. } | NodeType::Parameter { .. } | NodeType::Literal(_) | NodeType::External { .. } => (),
                ref op => return Err(AutogradError::NonDifferentiableOperation(op.clone())),
            }
        }

        let grad_outputs: Vec<NodeId> = with_respect_to
            .iter()
            .map(|id| self.get_or_create_zero_grad(*id))
            .collect();
        
        self.grad_asg.set_outputs(grad_outputs);
        
        let mut grad_initial_shapes = HashMap::new();
        for node in self.source_asg.nodes.values() {
            if let (Some(shape), Some(dtype)) = (&node.shape, &node.dtype) {
                 let external_node_name = format!("external_{}_{}", self.source_asg.id, node.id);
                 grad_initial_shapes.insert(external_node_name, (shape.clone(), *dtype));
            }
        }
        crate::analysis::shape_inference::ShapeInference::run(&mut self.grad_asg, &grad_initial_shapes)
            .map_err(|_e| AutogradError::MissingShapeInfo(0))?;

        Ok(self.grad_asg)
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
        let node = self.source_asg.nodes.get(&node_id).ok_or(AutogradError::NodeNotFound(node_id))?;

        let inputs = match &node.node_type {
            NodeType::Add(a, b) | NodeType::Subtract(a, b) | NodeType::Multiply(a, b) | NodeType::Divide(a, b) |
            NodeType::MatrixMultiply(a, b) | NodeType::GreaterThan(a, b) | NodeType::Power(a, b) |
            NodeType::Broadcast(a, b) | NodeType::Reshape(a, b) => vec![*a, *b],
            NodeType::ReLU(a) | NodeType::Sum(a) | NodeType::Sigmoid(a) | NodeType::Softmax(a) |
            NodeType::Mean(a) | NodeType::Variance(a) | NodeType::Sqrt(a) | NodeType::Log(a) => vec![*a],
            NodeType::Transpose(a, _, _) => vec![*a],
            _ => vec![],
        };

        for input_id in inputs {
            self.build_sorted_graph(input_id, visited, sorted)?;
        }
        sorted.push(node_id);
        Ok(())
    }

    fn accumulate_grad(&mut self, node_id: NodeId, grad_to_add_id: NodeId) {
        if let Some(&existing_grad_id) = self.grad_map.get(&node_id) {
            let new_grad_id = self.grad_asg.add_node(None, NodeType::Add(existing_grad_id, grad_to_add_id));
            self.grad_map.insert(node_id, new_grad_id);
        } else {
            self.grad_map.insert(node_id, grad_to_add_id);
        }
    }

    fn get_or_create_zero_grad(&mut self, node_id: NodeId) -> NodeId {
        if let Some(&grad_id) = self.grad_map.get(&node_id) {
            return grad_id;
        }

        let source_node = self.source_asg.get_node(node_id).expect("Node not found in source graph for zero grad");
        let shape = source_node.shape.as_ref().expect("Shape info missing for zero grad");

        let zeros = ndarray::ArrayD::zeros(shape.clone());
        
        let zero_grad_id = self.grad_asg.add_node(
            Some(format!("zero_grad_for_{}", node_id)),
            NodeType::Literal(Value::Tensor(zeros)),
        );
        
        let node_mut = self.grad_asg.get_node_mut(zero_grad_id).unwrap();
        node_mut.shape = Some(shape.clone());
        node_mut.dtype = source_node.dtype;

        self.grad_map.insert(node_id, zero_grad_id);
        zero_grad_id
    }
}