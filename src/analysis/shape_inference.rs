//! Вывод форм (shape inference) и dtype — безопасная полноценная версия
//! Поддерживает форму-источник как Literal(Tensor), так и любой узел с уже известной shape.

use crate::asg::{Asg, AsgError, DType, NodeId, NodeType, Shape, Value};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum ShapeInferenceError {
    #[error("Ошибка графа: {0}")]
    AsgError(#[from] AsgError),
    #[error("Несовместимые формы для операции '{op}': {shape1:?} и {shape2:?}")]
    IncompatibleShapes { op: String, shape1: Shape, shape2: Shape },
    #[error("Нет информации о форме для узла {0}")]
    MissingShapeInfo(NodeId),
    #[error("Для входа/параметра '{0}' не указана исходная форма")]
    MissingInitialShape(String),
    #[error("Неверный ранк (ожидалось {expected}, получено {actual}) для узла {node_id}")]
    InvalidRank { node_id: NodeId, expected: usize, actual: usize },
    #[error("Узел {0} должен быть Literal")]
    ExpectedLiteral(NodeId),
}

pub struct ShapeInference<'a> {
    asg: &'a mut Asg,
    cache: HashMap<NodeId, (Shape, DType)>,
}

impl<'a> ShapeInference<'a> {
    pub fn new(asg: &'a mut Asg) -> Self {
        Self { asg, cache: HashMap::new() }
    }

    fn get_shape_dtype(&self, id: NodeId) -> Result<(Shape, DType), ShapeInferenceError> {
        let node = self.asg.get_node(id)?;
        let shape = node.shape.clone().ok_or(ShapeInferenceError::MissingShapeInfo(id))?;
        let dtype = node.dtype.ok_or(ShapeInferenceError::MissingShapeInfo(id))?;
        Ok((shape, dtype))
    }

    fn try_shape_dtype(&self, id: NodeId) -> Option<(Shape, DType)> {
        let node = self.asg.get_node(id).ok()?;
        Some((node.shape.clone()?, node.dtype?))
    }

    /// Унифицированное чтение «целевой формы» из узла:
    /// - если Literal(Tensor) — читаем из данных;
    /// - иначе берём node.shape, если уже известна.
    fn target_shape_from_node(&self, id: NodeId) -> Result<Vec<usize>, ShapeInferenceError> {
        let node = self.asg.get_node(id)?;
        match &node.node_type {
            NodeType::Literal(Value::Tensor(arr)) => Ok(arr.shape().to_vec()),
            _ => node
                .shape
                .clone()
                .ok_or(ShapeInferenceError::MissingShapeInfo(id)),
        }
    }

    fn broadcast_shape(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
        let mut result = Vec::new();
        let mut i = a.len() as isize - 1;
        let mut j = b.len() as isize - 1;
        while i >= 0 || j >= 0 {
            let da = if i >= 0 { a[i as usize] } else { 1 };
            let db = if j >= 0 { b[j as usize] } else { 1 };
            if da == db || da == 1 || db == 1 { result.push(da.max(db)); } else { return None; }
            i -= 1; j -= 1;
        }
        result.reverse();
        Some(result)
    }

    /// Проход по узлам по возрастанию id; пишет shape/dtype обратно в граф без `unsafe`.
    /// «Мягкие» входы/параметры: отсутствие формы не считается ошибкой (формы могут быть выведены позже).
    pub fn infer(mut self) -> Result<HashMap<NodeId, (Shape, DType)>, ShapeInferenceError> {
        let node_count = self.asg.nodes.len();
        for id in 0..node_count {
            let node = self.asg.get_node(id)?;

            // Мягкая обработка Input/Parameter: если форма/тип не выставлены — пропускаем без ошибки.
            if let NodeType::Input { .. } | NodeType::Parameter { .. } = node.node_type {
                if node.shape.is_none() || node.dtype.is_none() {
                    continue;
                }
            }

            // Строго требуем форму/тип только для Literal/External
            if let NodeType::Literal(_) | NodeType::External { .. } = node.node_type {
                if node.shape.is_none() || node.dtype.is_none() {
                    return Err(ShapeInferenceError::MissingShapeInfo(id));
                }
            }

            let result: Result<(Shape, DType), ShapeInferenceError> = match node.node_type {
                NodeType::Input { .. } | NodeType::Parameter { .. } => {
                    let shape = node.shape.clone().unwrap();
                    let dtype = node.dtype.unwrap();
                    Ok((shape, dtype))
                }
                NodeType::Literal(_) | NodeType::External { .. } => {
                    let shape = node.shape.clone().unwrap();
                    let dtype = node.dtype.unwrap();
                    Ok((shape, dtype))
                }

                NodeType::Add(a, b)
                | NodeType::Subtract(a, b)
                | NodeType::Multiply(a, b)
                | NodeType::Divide(a, b)
                | NodeType::Power(a, b)
                | NodeType::GreaterThan(a, b)
                | NodeType::Less(a, b)
                | NodeType::Equal(a, b)
                | NodeType::MatrixMultiply(a, b) => {
                    let lhs_opt = self.try_shape_dtype(a);
                    let rhs_opt = self.try_shape_dtype(b);

                    match (lhs_opt, rhs_opt) {
                        (Some((shape_a, dtype_a)), Some((shape_b, dtype_b))) => {
                            if let NodeType::MatrixMultiply(_, _) = node.node_type {
                                if shape_a.len() != 2 || shape_b.len() != 2 {
                                    return Err(ShapeInferenceError::InvalidRank { node_id: id, expected: 2, actual: shape_a.len().max(shape_b.len()) });
                                }
                                if shape_a[1] != shape_b[0] {
                                    return Err(ShapeInferenceError::IncompatibleShapes { op: "MatMul".into(), shape1: shape_a, shape2: shape_b });
                                }
                                Ok((vec![shape_a[0], shape_b[1]], dtype_a))
                            } else {
                                let out_shape = Self::broadcast_shape(&shape_a, &shape_b)
                                    .ok_or(ShapeInferenceError::IncompatibleShapes { op: "broadcast".into(), shape1: shape_a.clone(), shape2: shape_b.clone() })?;
                                let out_dtype = match (dtype_a, dtype_b) {
                                    (DType::F32, _) | (_, DType::F32) => DType::F32,
                                    (DType::I32, DType::I32) => DType::I32,
                                    _ => DType::F32,
                                };
                                Ok((out_shape, out_dtype))
                            }
                        }
                        (Some((shape, dtype)), None) | (None, Some((shape, dtype))) => Ok((shape, dtype)),
                        (None, None) => Err(ShapeInferenceError::MissingShapeInfo(id)),
                    }
                }

                NodeType::Negate(x)
                | NodeType::Exp(x)
                | NodeType::Sqrt(x)
                | NodeType::Log(x)
                | NodeType::ReLU(x)
                | NodeType::Sigmoid(x)
                | NodeType::Softmax(x) => {
                    if let Some(sd) = self.try_shape_dtype(x) { Ok(sd) } else { Err(ShapeInferenceError::MissingShapeInfo(x)) }
                }

                NodeType::Sum(x) => {
                    let dtype = self.try_shape_dtype(x).map(|(_, dt)| dt).unwrap_or(DType::F32);
                    Ok((vec![], dtype))
                }
                NodeType::Mean(x) | NodeType::Variance(x) => {
                    if let Some((mut shape, dtype)) = self.try_shape_dtype(x) {
                        if !shape.is_empty() { *shape.last_mut().unwrap() = 1; }
                        Ok((shape, dtype))
                    } else {
                        Err(ShapeInferenceError::MissingShapeInfo(x))
                    }
                }
                NodeType::MeanAxis(x, axis) => {
                    if let Some((mut shape, dtype)) = self.try_shape_dtype(x) {
                        if !shape.is_empty() {
                            let rank = shape.len();
                            let mut ax = if axis < 0 { (rank as isize + axis) as usize } else { axis as usize };
                            if ax >= rank { ax = rank - 1; }
                            shape[ax] = 1;
                        }
                        Ok((shape, dtype))
                    } else {
                        Err(ShapeInferenceError::MissingShapeInfo(x))
                    }
                }

                NodeType::Transpose(x, a1, a2) => {
                    if let Some((mut shape, dtype)) = self.try_shape_dtype(x) {
                        if !shape.is_empty() {
                            if a1 >= shape.len() || a2 >= shape.len() {
                                return Err(ShapeInferenceError::InvalidRank { node_id: id, expected: shape.len(), actual: a1.max(a2) + 1 });
                            }
                            shape.swap(a1, a2);
                        }
                        Ok((shape, dtype))
                    } else {
                        Err(ShapeInferenceError::MissingShapeInfo(x))
                    }
                }

                // === Важно: целевая форма может приходить не только как Literal ===
                NodeType::Reshape(x, shape_node) => {
                    let new_shape = self.target_shape_from_node(shape_node)?;
                    let (_, dtype) = self.try_shape_dtype(x).unwrap_or((new_shape.clone(), DType::F32));
                    Ok((new_shape, dtype))
                }

                NodeType::Broadcast(x, target) => {
                    let target_shape = self.target_shape_from_node(target)?;
                    let (src_shape, src_dtype) = if let Some(sd) = self.try_shape_dtype(x) {
                        sd
                    } else {
                        (target_shape.clone(), DType::F32)
                    };
                    let _ = Self::broadcast_shape(&src_shape, &target_shape)
                        .ok_or(ShapeInferenceError::IncompatibleShapes { op: "broadcast".into(), shape1: src_shape.clone(), shape2: target_shape.clone() })?;
                    Ok((target_shape, src_dtype))
                }

                NodeType::ReduceSumTo(x, target_shape_node) => {
                    let target_shape = self.target_shape_from_node(target_shape_node)?;
                    let (src_shape, src_dtype) = if let Some(sd) = self.try_shape_dtype(x) {
                        sd
                    } else {
                        (target_shape.clone(), DType::F32)
                    };
                    let _ = Self::broadcast_shape(&src_shape, &target_shape)
                        .ok_or(ShapeInferenceError::IncompatibleShapes { op: "reduce_to".into(), shape1: src_shape.clone(), shape2: target_shape.clone() })?;
                    Ok((target_shape, src_dtype))
                }

                NodeType::MaxPool2d { input, kernel_size, stride } => {
                    if let Some((in_shape, dtype)) = self.try_shape_dtype(input) {
                        if in_shape.len() != 4 {
                            return Err(ShapeInferenceError::InvalidRank { node_id: input, expected: 4, actual: in_shape.len() });
                        }
                        let (n, c, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
                        let out_h = (h - kernel_size.0) / stride.0 + 1;
                        let out_w = (w - kernel_size.1) / stride.1 + 1;
                        Ok((vec![n, c, out_h, out_w], dtype))
                    } else {
                        Err(ShapeInferenceError::MissingShapeInfo(input))
                    }
                }

                NodeType::MaxUnpool2d { original_input, .. } => {
                    if let Some((in_shape, dtype)) = self.try_shape_dtype(original_input) {
                        Ok((in_shape, dtype))
                    } else {
                        Err(ShapeInferenceError::MissingShapeInfo(original_input))
                    }
                }

                NodeType::If { .. } | NodeType::ForLoop { .. } => Ok((vec![], DType::F32)),
            };

            let (shape, dtype) = result?;
            self.cache.insert(id, (shape.clone(), dtype));
            let node_mut = self.asg.get_node_mut(id)?;
            node_mut.shape = Some(shape);
            node_mut.dtype = Some(dtype);
        }
        Ok(self.cache)
    }

    /// Тривиальная «топологическая сортировка» по возрастанию id.
    pub fn topological_sort(asg: &Asg) -> Result<Vec<NodeId>, ShapeInferenceError> {
        let mut ids: Vec<NodeId> = asg.nodes.keys().cloned().collect();
        ids.sort_unstable();
        Ok(ids)
    }

    /// Проставляет `initial_shapes` для входов/параметров (если есть) и запускает инференс.
    /// Отсутствующие формы для Input/Parameter — не ошибка.
    pub fn run(asg: &mut Asg, initial_shapes: &HashMap<String, (Shape, DType)>) -> Result<(), ShapeInferenceError> {
        for id in 0..asg.nodes.len() {
            let node = asg.get_node(id)?;
            match &node.node_type {
                NodeType::Input { name } | NodeType::Parameter { name } => {
                    if let Some((shape, dtype)) = initial_shapes.get(name) {
                        let node_mut = asg.get_node_mut(id)?;
                        node_mut.shape = Some(shape.clone());
                        node_mut.dtype = Some(*dtype);
                    }
                }
                _ => {}
            }
        }
        let _ = ShapeInference::new(asg).infer()?;
        Ok(())
    }
}
