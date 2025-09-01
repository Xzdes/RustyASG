//! Модуль для автоматического дифференцирования (Autograd) на основе ASG.
//!
//! Основная идея этого модуля - преобразование графа в граф (Graph-to-Graph).
//! Мы берем ASG, который вычисляет некоторую функцию (например, потери), и
//! генерируем новый ASG, который вычисляет градиенты этой функции по
//! отношению к заданным переменным (например, весам модели).

use crate::asg::{Asg, NodeId, NodeType, Value};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum AutogradError {
    #[error("Узел {0} не найден в графе")]
    NodeNotFound(NodeId),
    #[error("Невозможно дифференцировать операцию {0:?}")]
    NonDifferentiableOperation(NodeType),
    #[error("Shape-информация отсутствует в исходном графе для узла {0}, что необходимо для Autograd")]
    MissingShapeInfo(NodeId),
}

/// Структура, отвечающая за генерацию градиента.
pub struct Gradients {
    /// Исходный граф, для которого мы вычисляем градиенты.
    /// **Ожидается, что Shape Inference для этого графа уже был выполнен.**
    source_asg: Asg,
    /// Новый граф, который будет содержать вычисления градиентов.
    grad_asg: Asg,
    /// Отображение ID узла из исходного графа в ID узла его градиента в новом графе.
    /// `grad_map[node_id] = grad_node_id`
    grad_map: HashMap<NodeId, NodeId>,
}

impl Gradients {
    /// Создает новый генератор градиентов.
    pub fn new(source_asg: Asg) -> Self {
        let grad_asg_id = source_asg.id + 1; // Простой способ получить новый ID
        Self {
            source_asg,
            grad_asg: Asg::new(grad_asg_id, Some("grad_graph".to_string())),
            grad_map: HashMap::new(),
        }
    }

    /// "Импортирует" узел из исходного графа в граф градиентов,
    /// создавая ссылку `External`.
    fn import_node(&mut self, source_node_id: NodeId) -> NodeId {
        self.grad_asg.add_node(
            None,
            NodeType::External {
                source_asg_id: self.source_asg.id,
                source_node_id,
            },
        )
    }

    /// Главная публичная функция. Строит граф, вычисляющий градиенты
    /// для `target_node_id` по отношению ко всем узлам в `with_respect_to`.
    pub fn build(
        mut self,
        target_node_id: NodeId,
        with_respect_to: &[NodeId],
    ) -> Result<Asg, AutogradError> {
        let sorted_nodes = self.topological_sort(target_node_id)?;

        // dy/dy = 1
        let grad_of_target_node_id = self.grad_asg.add_node(
            None,
            NodeType::Literal(Value::Tensor(ndarray::arr0(1.0f32).into_dyn())),
        );
        self.grad_map
            .insert(target_node_id, grad_of_target_node_id);

        // Обратный проход по графу
        for &node_id in sorted_nodes.iter().rev() {
            let node = self
                .source_asg
                .nodes
                .get(&node_id)
                .ok_or(AutogradError::NodeNotFound(node_id))?;
            
            let node_type = node.node_type.clone();

            let upstream_grad_id = self.get_or_create_zero_grad(node_id);

            match node_type {
                NodeType::Add(lhs_id, rhs_id) => {
                    self.accumulate_grad(lhs_id, upstream_grad_id);
                    self.accumulate_grad(rhs_id, upstream_grad_id);
                }
                NodeType::Subtract(lhs_id, rhs_id) => {
                    self.accumulate_grad(lhs_id, upstream_grad_id);
                    let neg_one = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(-1.0f32).into_dyn())));
                    let neg_upstream_grad = self.grad_asg.add_node(None, NodeType::Multiply(neg_one, upstream_grad_id));
                    self.accumulate_grad(rhs_id, neg_upstream_grad);
                }
                NodeType::Multiply(a_id, b_id) => {
                    let imported_b = self.import_node(b_id);
                    let grad_a = self.grad_asg.add_node(None, NodeType::Multiply(upstream_grad_id, imported_b));
                    self.accumulate_grad(a_id, grad_a);
                    
                    let imported_a = self.import_node(a_id);
                    let grad_b = self.grad_asg.add_node(None, NodeType::Multiply(upstream_grad_id, imported_a));
                    self.accumulate_grad(b_id, grad_b);
                }
                NodeType::Divide(a_id, b_id) => {
                    let imported_a = self.import_node(a_id);
                    let imported_b = self.import_node(b_id);
                    let b_squared = self.grad_asg.add_node(None, NodeType::Multiply(imported_b, imported_b));
                    let grad_a = self.grad_asg.add_node(None, NodeType::Divide(upstream_grad_id, imported_b));
                    self.accumulate_grad(a_id, grad_a);

                    let neg_one = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(-1.0f32).into_dyn())));
                    let term1 = self.grad_asg.add_node(None, NodeType::Multiply(neg_one, upstream_grad_id));
                    let term2 = self.grad_asg.add_node(None, NodeType::Multiply(term1, imported_a));
                    let grad_b = self.grad_asg.add_node(None, NodeType::Divide(term2, b_squared));
                    self.accumulate_grad(b_id, grad_b);
                }
                NodeType::MatrixMultiply(a_id, b_id) => {
                    let imported_b = self.import_node(b_id);
                    let b_transposed_id = self.grad_asg.add_node(None, NodeType::Transpose(imported_b, 0, 1));
                    let grad_a = self.grad_asg.add_node(None, NodeType::MatrixMultiply(upstream_grad_id, b_transposed_id));
                    self.accumulate_grad(a_id, grad_a);

                    let imported_a = self.import_node(a_id);
                    let a_transposed_id = self.grad_asg.add_node(None, NodeType::Transpose(imported_a, 0, 1));
                    let grad_b = self.grad_asg.add_node(None, NodeType::MatrixMultiply(a_transposed_id, upstream_grad_id));
                    self.accumulate_grad(b_id, grad_b);
                }
                NodeType::Sum(x_id) => {
                    let imported_x = self.import_node(x_id);
                    let grad_x = self.grad_asg.add_node(None, NodeType::Broadcast(upstream_grad_id, imported_x));
                    self.accumulate_grad(x_id, grad_x);
                }
                NodeType::Mean(x_id) => {
                    let x_node = self.source_asg.nodes.get(&x_id).ok_or(AutogradError::NodeNotFound(x_id))?;
                    let x_shape = x_node.shape.as_ref().ok_or(AutogradError::MissingShapeInfo(x_id))?;
                    let n_val = *x_shape.last().unwrap_or(&1) as f32; // Получаем размерность динамически!
                    
                    let n = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(n_val).into_dyn())));
                    let imported_x = self.import_node(x_id);
                    let broadcasted_grad = self.grad_asg.add_node(None, NodeType::Broadcast(upstream_grad_id, imported_x));
                    let final_grad = self.grad_asg.add_node(None, NodeType::Divide(broadcasted_grad, n));
                    self.accumulate_grad(x_id, final_grad);
                }
                NodeType::ReLU(x_id) => {
                    let imported_x = self.import_node(x_id);
                    let zero = self.grad_asg.add_node(None, NodeType::Literal(Value::Tensor(ndarray::arr0(0.0f32).into_dyn())));
                    let step_mask_id = self.grad_asg.add_node(None, NodeType::GreaterThan(imported_x, zero));
                    let local_grad_id = self.grad_asg.add_node(None, NodeType::Multiply(upstream_grad_id, step_mask_id));
                    self.accumulate_grad(x_id, local_grad_id);
                }
                NodeType::Transpose(x_id, axis1, axis2) => {
                    // Градиент транспонирования - это транспонирование градиента
                    let grad_x = self.grad_asg.add_node(None, NodeType::Transpose(upstream_grad_id, axis1, axis2));
                    self.accumulate_grad(x_id, grad_x);
                }
                NodeType::Reshape(x_id, _) => {
                    // Градиент reshape - это reshape градиента в исходную форму
                    let imported_x = self.import_node(x_id);
                    let grad_x = self.grad_asg.add_node(None, NodeType::Reshape(upstream_grad_id, imported_x));
                    self.accumulate_grad(x_id, grad_x);
                }

                // Узлы, которые не имеют входов в графе и являются "листьями"
                NodeType::Input { .. } | NodeType::Parameter { .. } | NodeType::Literal(_) | NodeType::External { .. } => (),

                // Операции, для которых градиент пока не реализован
                ref op => return Err(AutogradError::NonDifferentiableOperation(op.clone())),
            }
        }

        // Создаем выходы для графа градиентов
        let grad_outputs: Vec<NodeId> = with_respect_to
            .iter()
            .map(|id| self.get_or_create_zero_grad(*id))
            .collect();
        
        self.grad_asg.set_outputs(grad_outputs);

        Ok(self.grad_asg)
    }

    // --- Вспомогательные функции ---

    fn topological_sort(&self, start_node_id: NodeId) -> Result<Vec<NodeId>, AutogradError> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        self.build_sorted_graph(start_node_id, &mut visited, &mut sorted)?;
        Ok(sorted)
    }
    
    fn build_sorted_graph(&self, node_id: NodeId, visited: &mut HashSet<NodeId>, sorted: &mut Vec<NodeId>) -> Result<(), AutogradError> {
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
            if !visited.contains(&input_id) {
                self.build_sorted_graph(input_id, visited, sorted)?;
            }
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
        let zero_grad_id = self.grad_asg.add_node(
            None,
            NodeType::Literal(Value::Tensor(ndarray::arr0(0.0f32).into_dyn())),
        );
        self.grad_map.insert(node_id, zero_grad_id);
        zero_grad_id
    }
}