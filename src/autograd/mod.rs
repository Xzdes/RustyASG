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
}

/// Структура, отвечающая за генерацию градиента.
pub struct Gradients {
    /// Исходный граф, для которого мы вычисляем градиенты.
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
        // Мы могли бы кэшировать импортированные узлы, но для простоты пока создаем новый каждый раз.
        self.grad_asg.add_node(
            None,
            NodeType::External {
                source_asg_id: self.source_asg.id,
                source_node_id,
            },
        )
    }

    /// Главная публичная функция. Строит граф градиентов.
    ///
    /// # Аргументы
    /// * `target_node_id` - ID узла, по отношению к которому мы дифференцируем (обычно `loss`).
    /// * `with_respect_to` - Список ID узлов, для которых нужно вычислить градиент (обычно параметры модели).
    ///
    /// # Возвращает
    /// Новый ASG, выходы которого - это градиенты для узлов из `with_respect_to`.
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
        self.grad_map
            .insert(target_node_id, grad_of_target_node_id);

        for &node_id in sorted_nodes.iter().rev() {
            let node_type = {
                let node = self
                    .source_asg
                    .nodes
                    .get(&node_id)
                    .ok_or(AutogradError::NodeNotFound(node_id))?;
                node.node_type.clone()
            };

            let upstream_grad_id = self.get_or_create_zero_grad(node_id);

            match node_type {
                NodeType::Add(lhs_id, rhs_id) => {
                    self.accumulate_grad(lhs_id, upstream_grad_id);
                    self.accumulate_grad(rhs_id, upstream_grad_id);
                }

                NodeType::Subtract(lhs_id, rhs_id) => {
                    // grad(lhs) = upstream_grad
                    self.accumulate_grad(lhs_id, upstream_grad_id);
                    // grad(rhs) = -1 * upstream_grad
                    let neg_one = self.grad_asg.add_node(
                        None,
                        NodeType::Literal(Value::Tensor(ndarray::arr0(-1.0f32).into_dyn())),
                    );
                    let neg_upstream_grad = self
                        .grad_asg
                        .add_node(None, NodeType::Multiply(neg_one, upstream_grad_id));
                    self.accumulate_grad(rhs_id, neg_upstream_grad);
                }

                NodeType::Multiply(a_id, b_id) => {
                    let imported_a = self.import_node(a_id);
                    let imported_b = self.import_node(b_id);
                    // grad(A) = upstream_grad * B
                    let grad_a = self
                        .grad_asg
                        .add_node(None, NodeType::Multiply(upstream_grad_id, imported_b));
                    self.accumulate_grad(a_id, grad_a);
                    // grad(B) = upstream_grad * A
                    let grad_b = self
                        .grad_asg
                        .add_node(None, NodeType::Multiply(upstream_grad_id, imported_a));
                    self.accumulate_grad(b_id, grad_b);
                }

                NodeType::MatrixMultiply(a_id, b_id) => {
                    let imported_a = self.import_node(a_id);
                    let imported_b = self.import_node(b_id);

                    // grad(A) += upstream_grad @ B^T
                    let b_transposed_id =
                        self.grad_asg
                            .add_node(None, NodeType::Transpose(imported_b, 0, 1));
                    let grad_a = self.grad_asg.add_node(
                        None,
                        NodeType::MatrixMultiply(upstream_grad_id, b_transposed_id),
                    );
                    self.accumulate_grad(a_id, grad_a);

                    // grad(B) += A^T @ upstream_grad
                    let a_transposed_id =
                        self.grad_asg
                            .add_node(None, NodeType::Transpose(imported_a, 0, 1));
                    let grad_b = self.grad_asg.add_node(
                        None,
                        NodeType::MatrixMultiply(a_transposed_id, upstream_grad_id),
                    );
                    self.accumulate_grad(b_id, grad_b);
                }

                NodeType::Power(base_id, power_id) => {
                    let imported_base = self.import_node(base_id);
                    let imported_power = self.import_node(power_id);
                    // grad(base) = upstream_grad * power * base^(power-1)
                    let one = self.grad_asg.add_node(
                        None,
                        NodeType::Literal(Value::Tensor(ndarray::arr0(1.0f32).into_dyn())),
                    );
                    let power_minus_1 = self
                        .grad_asg
                        .add_node(None, NodeType::Subtract(imported_power, one));
                    let base_pow = self
                        .grad_asg
                        .add_node(None, NodeType::Power(imported_base, power_minus_1));
                    let term1 = self
                        .grad_asg
                        .add_node(None, NodeType::Multiply(upstream_grad_id, imported_power));
                    let grad_base =
                        self.grad_asg
                            .add_node(None, NodeType::Multiply(term1, base_pow));
                    self.accumulate_grad(base_id, grad_base);
                }

                NodeType::ReLU(x_id) => {
                    let imported_x = self.import_node(x_id);
                    // grad(X) += upstream_grad * Step(X)
                    let zero_tensor = ndarray::arr0(0.0f32).into_dyn();
                    let zero_id = self
                        .grad_asg
                        .add_node(None, NodeType::Literal(Value::Tensor(zero_tensor)));
                    let step_mask_id = self
                        .grad_asg
                        .add_node(None, NodeType::GreaterThan(imported_x, zero_id));
                    let local_grad_id = self
                        .grad_asg
                        .add_node(None, NodeType::Multiply(upstream_grad_id, step_mask_id));
                    self.accumulate_grad(x_id, local_grad_id);
                }

                NodeType::Sum(x_id) => {
                    let imported_x = self.import_node(x_id);
                    // grad(x) = broadcast(upstream_grad, shape_of(x))
                    let grad_x = self
                        .grad_asg
                        .add_node(None, NodeType::Broadcast(upstream_grad_id, imported_x));
                    self.accumulate_grad(x_id, grad_x);
                }

                // Листовые и недифференцируемые узлы
                NodeType::Input { .. }
                | NodeType::Parameter { .. }
                | NodeType::Literal(_)
                | NodeType::External { .. }
                | NodeType::Transpose(_, _, _)
                | NodeType::GreaterThan(_, _)
                | NodeType::Broadcast(_, _) => (),

                ref op => return Err(AutogradError::NonDifferentiableOperation(op.clone())),
            }
        }

        let grad_outputs: Vec<NodeId> = with_respect_to
            .iter()
            .map(|id| self.get_or_create_zero_grad(*id))
            .collect();

        if let Some(first_output) = grad_outputs.get(0) {
            self.grad_asg.set_output(*first_output);
        } else {
            let dummy_output = self
                .grad_asg
                .add_node(None, NodeType::Literal(Value::Unit));
            self.grad_asg.set_output(dummy_output);
        }

        Ok(self.grad_asg)
    }

    fn build_sorted_graph(
        &self,
        node_id: NodeId,
        visited: &mut HashSet<NodeId>,
        sorted: &mut Vec<NodeId>,
    ) -> Result<(), AutogradError> {
        visited.insert(node_id);
        let node = self
            .source_asg
            .nodes
            .get(&node_id)
            .ok_or(AutogradError::NodeNotFound(node_id))?;

        let inputs = match &node.node_type {
            NodeType::Add(a, b)
            | NodeType::Subtract(a, b)
            | NodeType::Multiply(a, b)
            | NodeType::MatrixMultiply(a, b)
            | NodeType::GreaterThan(a, b)
            | NodeType::Power(a, b)
            | NodeType::Broadcast(a, b) => vec![*a, *b],
            NodeType::ReLU(a) | NodeType::Sum(a) => vec![*a],
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

    fn topological_sort(&self, start_node_id: NodeId) -> Result<Vec<NodeId>, AutogradError> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        self.build_sorted_graph(start_node_id, &mut visited, &mut sorted)?;
        Ok(sorted)
    }

    fn accumulate_grad(&mut self, node_id: NodeId, grad_to_add_id: NodeId) {
        if let Some(&existing_grad_id) = self.grad_map.get(&node_id) {
            let new_grad_id = self
                .grad_asg
                .add_node(None, NodeType::Add(existing_grad_id, grad_to_add_id));
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