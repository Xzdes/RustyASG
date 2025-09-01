//! Модуль для вывода форм и типов данных (Shape Inference).
//!
//! Проходит по графу вычислений и для каждого узла определяет форму
//! и тип выходного тензора на основе форм его входов и типа операции.

use crate::asg::{Asg, AsgError, DType, Node, NodeId, NodeType, Shape, Value};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum ShapeInferenceError {
    #[error("Ошибка графа: {0}")]
    AsgError(#[from] AsgError),
    #[error("Несовместимые формы для операции '{op}': {shape1:?} и {shape2:?}")]
    IncompatibleShapes {
        op: String,
        shape1: Shape,
        shape2: Shape,
    },
    #[error("Информация о форме отсутствует для узла {0}, необходимого для вывода")]
    MissingShapeInfo(NodeId),
    #[error("Для входа или параметра с именем '{0}' не предоставлена информация о форме")]
    MissingInitialShape(String),
    #[error("Неверное количество измерений (ожидалось {expected}, получено {actual}) для узла {node_id}")]
    InvalidRank {
        node_id: NodeId,
        expected: usize,
        actual: usize,
    },
    #[error("Узел {0} должен быть константой (Literal) для вывода формы")]
    NotALiteral(NodeId),
}

type Result<T> = std::result::Result<T, ShapeInferenceError>;

/// Структура, выполняющая вывод форм для ASG.
pub struct ShapeInference;

impl ShapeInference {
    /// Запускает процесс вывода форм для графа.
    ///
    /// Модифицирует граф "на месте", заполняя поля `shape` и `dtype` для каждого узла.
    ///
    /// # Аргументы
    /// * `asg` - Изменяемая ссылка на граф, который нужно проанализировать.
    /// * `initial_shapes` - HashMap, предоставляющий формы и типы для всех
    ///   входных (`Input`) и параметрических (`Parameter`) узлов. Ключ - имя узла.
    pub fn run(asg: &mut Asg, initial_shapes: &HashMap<String, (Shape, DType)>) -> Result<()> {
        // Выполняем топологическую сортировку для всех выходных узлов.
        let sorted_nodes = Self::topological_sort(asg)?;

        for node_id in sorted_nodes {
            let mut node = asg.get_node(node_id)?.clone();

            // Вычисляем форму и тип для текущего узла
            let (shape, dtype) = Self::infer_node_shape(asg, &node, initial_shapes)?;

            // Обновляем узел в графе
            node.shape = Some(shape);
            node.dtype = Some(dtype);
            asg.nodes.insert(node_id, node);
        }

        Ok(())
    }

    /// Основная логика вывода формы для одного узла.
    fn infer_node_shape(
        asg: &Asg,
        node: &Node,
        initial_shapes: &HashMap<String, (Shape, DType)>,
    ) -> Result<(Shape, DType)> {
        match &node.node_type {
            // Узлы, форма которых определяется извне
            NodeType::Input { name } | NodeType::Parameter { name } => initial_shapes
                .get(name)
                .cloned()
                .ok_or(ShapeInferenceError::MissingInitialShape(name.clone())),

            // Форма определяется значением литерала
            NodeType::Literal(value) => match value {
                Value::Tensor(arr) => Ok((arr.shape().to_vec(), DType::F32)),
                Value::Integer(_) => Ok((vec![], DType::I64)),
                Value::Boolean(_) => Ok((vec![], DType::Bool)),
                _ => Ok((vec![], DType::F32)), // Для Unit, Text и т.д.
            },

            // Бинарные поэлементные операции
            NodeType::Add(l, r)
            | NodeType::Subtract(l, r)
            | NodeType::Multiply(l, r)
            | NodeType::Divide(l, r) => {
                let (ls, ld) = Self::get_shape_dtype(asg, *l)?;
                let (rs, rd) = Self::get_shape_dtype(asg, *r)?;
                // TODO: Реализовать полные правила broadcasting. Пока требуем равенства.
                if ls != rs {
                    return Err(ShapeInferenceError::IncompatibleShapes {
                        op: "element-wise".to_string(),
                        shape1: ls,
                        shape2: rs,
                    });
                }
                Ok((ls, ld)) // Тип наследуется от левого операнда
            }

            // Матричное умножение
            NodeType::MatrixMultiply(l, r) => {
                let (ls, ld) = Self::get_shape_dtype(asg, *l)?;
                let (rs, _) = Self::get_shape_dtype(asg, *r)?;

                if ls.len() < 2 || rs.len() < 2 {
                    return Err(ShapeInferenceError::InvalidRank {
                        node_id: node.id,
                        expected: 2,
                        actual: ls.len().min(rs.len()),
                    });
                }

                let m = ls[ls.len() - 2];
                let k1 = ls[ls.len() - 1];
                let k2 = rs[rs.len() - 2];
                let n = rs[rs.len() - 1];

                if k1 != k2 {
                    return Err(ShapeInferenceError::IncompatibleShapes {
                        op: "MatrixMultiply".to_string(),
                        shape1: ls,
                        shape2: rs,
                    });
                }

                let mut out_shape = ls[..ls.len() - 2].to_vec();
                out_shape.push(m);
                out_shape.push(n);

                Ok((out_shape, ld))
            }

            // Унарные поэлементные операции
            NodeType::ReLU(id) | NodeType::Sigmoid(id) | NodeType::Sqrt(id) => {
                Self::get_shape_dtype(asg, *id)
            }

            // Операции редукции
            NodeType::Sum(_) => Ok((vec![], DType::F32)), // Сумма по всему тензору -> скаляр
            NodeType::Mean(id) | NodeType::Variance(id) => {
                let (shape, dtype) = Self::get_shape_dtype(asg, *id)?;
                if shape.is_empty() {
                    Ok((vec![], dtype)) // Редукция скаляра - скаляр
                } else {
                    // Редукция по последней оси
                    let new_shape = shape[..shape.len() - 1].to_vec();
                    Ok((new_shape, dtype))
                }
            }

            // Трансформации
            NodeType::Transpose(id, axis1, axis2) => {
                let (mut shape, dtype) = Self::get_shape_dtype(asg, *id)?;
                shape.swap(*axis1, *axis2);
                Ok((shape, dtype))
            }

            NodeType::Reshape(data_id, shape_id) => {
                let (_, dtype) = Self::get_shape_dtype(asg, *data_id)?;
                let shape_node = asg.get_node(*shape_id)?;
                if let NodeType::Literal(Value::Tensor(shape_tensor)) = &shape_node.node_type {
                    let new_shape: Shape = shape_tensor.iter().map(|&x| x as usize).collect();
                    Ok((new_shape, dtype))
                } else {
                    Err(ShapeInferenceError::NotALiteral(*shape_id))
                }
            }

            NodeType::Broadcast(source_id, target_id) => {
                let (_, dtype) = Self::get_shape_dtype(asg, *source_id)?;
                let (target_shape, _) = Self::get_shape_dtype(asg, *target_id)?;
                Ok((target_shape, dtype))
            }

            NodeType::Softmax(id) => Self::get_shape_dtype(asg, *id), // Softmax не меняет форму

            // TODO: Реализовать вывод для остальных узлов
            _ => {
                // Временная заглушка для нереализованных узлов
                Ok((vec![1], DType::F32))
            }
        }
    }

    /// Вспомогательная функция для получения уже вычисленной формы и типа узла.
    fn get_shape_dtype(asg: &Asg, node_id: NodeId) -> Result<(Shape, DType)> {
        let node = asg.get_node(node_id)?;
        match (&node.shape, &node.dtype) {
            (Some(s), Some(d)) => Ok((s.clone(), *d)),
            _ => Err(ShapeInferenceError::MissingShapeInfo(node_id)),
        }
    }

    /// Выполняет топологическую сортировку графа.
    /// Возвращает вектор ID узлов в порядке, пригодном для вычислений.
    pub fn topological_sort(asg: &Asg) -> Result<Vec<NodeId>> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        for output_id in &asg.outputs {
            Self::build_sorted_graph(*output_id, asg, &mut visited, &mut sorted)?;
        }
        Ok(sorted)
    }

    fn build_sorted_graph(
        node_id: NodeId,
        asg: &Asg,
        visited: &mut HashSet<NodeId>,
        sorted: &mut Vec<NodeId>,
    ) -> Result<()> {
        visited.insert(node_id);
        let node = asg.get_node(node_id)?;

        let inputs = match &node.node_type {
            NodeType::Add(a, b)
            | NodeType::Subtract(a, b)
            | NodeType::Multiply(a, b)
            | NodeType::Divide(a, b)
            | NodeType::MatrixMultiply(a, b)
            | NodeType::GreaterThan(a, b)
            | NodeType::Power(a, b)
            | NodeType::Broadcast(a, b)
            | NodeType::Reshape(a, b) => vec![*a, *b],

            NodeType::ReLU(a)
            | NodeType::Sum(a)
            | NodeType::Sigmoid(a)
            | NodeType::Softmax(a)
            | NodeType::Mean(a)
            | NodeType::Variance(a)
            | NodeType::Sqrt(a)
            | NodeType::Log(a) => vec![*a],

            NodeType::Transpose(a, _, _) => vec![*a],
            _ => vec![], // Для Input, Parameter, Literal и т.д.
        };

        for input_id in inputs {
            if !visited.contains(&input_id) {
                Self::build_sorted_graph(input_id, asg, visited, sorted)?;
            }
        }
        sorted.push(node_id);
        Ok(())
    }
}