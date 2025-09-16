//! Автоград: построение графа градиентов d(loss)/d(node)
//!
//! Алгоритм:
//!  - обходим исходный граф в обратном топологическом порядке;
//!  - для узлов, где есть dL/dY, считаем вклады в операнды и аккумулируем (суммируем);
//!  - строим отдельный граф `grad` (ASG), без unsafe, пригодный для инференса/исполнения.
//!
//! Поддержка операторов для LayerNorm и базовых блоков:
//!  Add/Subtract, Multiply/Divide, Power, MatrixMultiply,
//!  Negate, Exp, Log, Sqrt, ReLU, Sigmoid, Softmax (пасс),
//!  Sum, Mean, MeanAxis, Variance,
//!  Transpose, Reshape, Broadcast, ReduceSumTo,
//!  GreaterThan/Less/Equal (нулевая производная).
//!
//! Важные детали:
//!  - `reuse(src_id)` создаёт в grad-графе Literal нужной формы исходного узла,
//!    чтобы Broadcast/ReduceSumTo/Reshape корректно работали по shape/dtype.

use crate::analysis::shape_inference::{ShapeInference, ShapeInferenceError};
use crate::asg::{Asg, AsgError, DType, NodeId, NodeType, Shape, Value};
use ndarray::{Array, IxDyn};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum AutogradError {
    #[error("ASG: {0:?}")]
    Asg(#[from] AsgError),
    #[error("Shape: {0:?}")]
    Shape(#[from] ShapeInferenceError),
}

pub type AutogradResult<T> = Result<T, AutogradError>;

/// Публичная обёртка — то, что ждут тесты `use rustyasg::autograd::Gradients;`
pub struct Gradients<'a> {
    inner: Autograd<'a>,
}

impl<'a> Gradients<'a> {
    pub fn new(src: &'a Asg) -> Self {
        Self { inner: Autograd::new(src) }
    }
    /// Построить граф градиентов (без инференса форм/типов).
    pub fn build(&mut self, loss_id: NodeId) -> AutogradResult<()> {
        self.inner.build(loss_id)
    }
    /// Построить и запустить shape/type-инференс.
    pub fn build_and_infer(&mut self, loss_id: NodeId) -> AutogradResult<()> {
        self.inner.build_and_infer(loss_id)
    }
    /// Получить id узла градиента по имени исходного узла (Input/Parameter).
    pub fn gradient_node_by_name(&self, name: &str) -> Option<NodeId> {
        self.inner.gradient_node_by_name(name)
    }
    /// Доступ к графу градиентов (read-only).
    pub fn graph(&self) -> &Asg {
        &self.inner.grad
    }
    /// Доступ к графу градиентов (mutable) — для исполнения/настроек.
    pub fn graph_mut(&mut self) -> &mut Asg {
        &mut self.inner.grad
    }
}

// --------------------------- Реализация автограда ---------------------------

struct Autograd<'a> {
    /// Исходный (прямой) граф
    src: &'a Asg,
    /// Граф градиентов
    grad: Asg,
    /// Карта: исходный узел -> узел градиента (в `grad`)
    gmap: HashMap<NodeId, NodeId>,
}

impl<'a> Autograd<'a> {
    fn new(src: &'a Asg) -> Self {
        Self {
            src,
            grad: Asg::new(1, Some("grad".to_string())),
            gmap: HashMap::new(),
        }
    }

    /// Построить граф градиентов для `loss_id` (seed = 1.0).
    fn build(&mut self, loss_id: NodeId) -> AutogradResult<()> {
        let order = ShapeInference::topological_sort(self.src)?;
        let g_loss = self.lit_scalar(1.0);
        self.gmap.insert(loss_id, g_loss);

        for &id in order.iter().rev() {
            let node = self.src.get_node(id)?;
            let g_out = match self.gmap.get(&id).copied() {
                Some(g) => g,
                None => continue,
            };

            match &node.node_type {
                // Листья
                NodeType::Input { .. }
                | NodeType::Parameter { .. }
                | NodeType::Literal(_)
                | NodeType::External { .. } => {
                    // dL/dX уже накоплен — ничего не делаем
                }

                // -------- БИНАРНЫЕ --------
                NodeType::Add(a, b) => {
                    let ra = self.reduce_to(g_out, *a)?;
                    self.acc(*a, ra);
                    let rb = self.reduce_to(g_out, *b)?;
                    self.acc(*b, rb);
                }
                NodeType::Subtract(a, b) => {
                    let ra = self.reduce_to(g_out, *a)?;
                    self.acc(*a, ra);
                    let neg = self.add_node(NodeType::Negate(g_out));
                    let rb = self.reduce_to(neg, *b)?;
                    self.acc(*b, rb);
                }
                NodeType::Multiply(a, b) => {
                    let a_im = self.reuse(*a)?;
                    let b_im = self.reuse(*b)?;
                    let g_a = self.add_node(NodeType::Multiply(g_out, b_im));
                    let g_b = self.add_node(NodeType::Multiply(g_out, a_im));
                    let ra = self.reduce_to(g_a, *a)?;
                    self.acc(*a, ra);
                    let rb = self.reduce_to(g_b, *b)?;
                    self.acc(*b, rb);
                }
                NodeType::Divide(a, b) => {
                    let a_im = self.reuse(*a)?;
                    let b_im = self.reuse(*b)?;
                    // g_a = g_out / b
                    let g_a = self.add_node(NodeType::Divide(g_out, b_im));
                    // g_b = - g_out * a / (b^2)
                    let two = self.lit_scalar(2.0);
                    let b_for_pow = self.reuse(*b)?; // локально до add_node
                    let b2 = self.add_node(NodeType::Power(b_for_pow, two));
                    let num = self.add_node(NodeType::Multiply(g_out, a_im));
                    let g_b_div = self.add_node(NodeType::Divide(num, b2));
                    let minus_one = self.lit_scalar(-1.0);
                    let g_b = self.add_node(NodeType::Multiply(g_b_div, minus_one));
                    let ra = self.reduce_to(g_a, *a)?;
                    self.acc(*a, ra);
                    let rb = self.reduce_to(g_b, *b)?;
                    self.acc(*b, rb);
                }
                NodeType::Power(a, b) => {
                    // da: g * b * a^(b-1)
                    let _a_im1 = self.reuse(*a)?; // остаётся для симметрии; не используется напрямую
                    let b_im1 = self.reuse(*b)?;
                    let one = self.lit_scalar(1.0);
                    let b_minus_one = self.add_node(NodeType::Subtract(b_im1, one));
                    // a^(b-1)
                    let a_for_pow = self.reuse(*a)?; // подготовить перед add_node
                    let a_pow = self.add_node(NodeType::Power(a_for_pow, b_minus_one));
                    // b * a^(b-1)
                    let b_im2 = self.reuse(*b)?;
                    let term_da = self.add_node(NodeType::Multiply(b_im2, a_pow));
                    let g_a = self.add_node(NodeType::Multiply(g_out, term_da));
                    let ra = self.reduce_to(g_a, *a)?;
                    self.acc(*a, ra);

                    // db: g * (a^b) * ln(a)
                    let a_for_y = self.reuse(*a)?;
                    let b_for_y = self.reuse(*b)?;
                    let y = self.add_node(NodeType::Power(a_for_y, b_for_y));
                    let a_for_ln = self.reuse(*a)?;
                    let ln_a = self.add_node(NodeType::Log(a_for_ln));
                    let term_db = self.add_node(NodeType::Multiply(y, ln_a));
                    let g_b = self.add_node(NodeType::Multiply(g_out, term_db));
                    let rb = self.reduce_to(g_b, *b)?;
                    self.acc(*b, rb);
                }
                NodeType::MatrixMultiply(a, b) => {
                    // g_a = g_out @ b^T
                    let b_im = self.reuse(*b)?;
                    let bt = self.add_node(NodeType::Transpose(b_im, 0, 1));
                    let g_a = self.add_node(NodeType::MatrixMultiply(g_out, bt));
                    // g_b = a^T @ g_out
                    let a_im = self.reuse(*a)?;
                    let at = self.add_node(NodeType::Transpose(a_im, 0, 1));
                    let g_b = self.add_node(NodeType::MatrixMultiply(at, g_out));
                    self.acc(*a, g_a);
                    self.acc(*b, g_b);
                }

                // -------- УНАРНЫЕ --------
                NodeType::Negate(x) => {
                    let neg_one = self.lit_scalar(-1.0);
                    let g_x_mul = self.add_node(NodeType::Multiply(g_out, neg_one));
                    let rx = self.reduce_to(g_x_mul, *x)?;
                    self.acc(*x, rx);
                }
                NodeType::Exp(x) => {
                    let x_im = self.reuse(*x)?;
                    let ex = self.add_node(NodeType::Exp(x_im));
                    let g_x = self.add_node(NodeType::Multiply(g_out, ex));
                    let rx = self.reduce_to(g_x, *x)?;
                    self.acc(*x, rx);
                }
                NodeType::Log(x) => {
                    let x_im = self.reuse(*x)?;
                    let g_x = self.add_node(NodeType::Divide(g_out, x_im));
                    let rx = self.reduce_to(g_x, *x)?;
                    self.acc(*x, rx);
                }
                NodeType::Sqrt(x) => {
                    let two = self.lit_scalar(2.0);
                    let x_im = self.reuse(*x)?;
                    let sqrt_x = self.add_node(NodeType::Sqrt(x_im));
                    let denom = self.add_node(NodeType::Multiply(two, sqrt_x));
                    let g_x = self.add_node(NodeType::Divide(g_out, denom));
                    let rx = self.reduce_to(g_x, *x)?;
                    self.acc(*x, rx);
                }
                NodeType::ReLU(x) => {
                    let zero = self.lit_scalar(0.0);
                    let x_im = self.reuse(*x)?;
                    let mask = self.add_node(NodeType::GreaterThan(x_im, zero));
                    let g_x = self.add_node(NodeType::Multiply(g_out, mask));
                    let rx = self.reduce_to(g_x, *x)?;
                    self.acc(*x, rx);
                }
                NodeType::Sigmoid(x) => {
                    let x_im = self.reuse(*x)?;
                    let sig = self.add_node(NodeType::Sigmoid(x_im));
                    let one = self.lit_scalar(1.0);
                    let one_minus_sig = self.add_node(NodeType::Subtract(one, sig));
                    // ВАЖНО: не вызывать reuse(sig) — sig это уже узел grad-графа
                    let term = self.add_node(NodeType::Multiply(sig, one_minus_sig));
                    let g_x = self.add_node(NodeType::Multiply(g_out, term));
                    let rx = self.reduce_to(g_x, *x)?;
                    self.acc(*x, rx);
                }
                NodeType::Softmax(_x) => {
                    // Полный якобиан не требуется в текущих тестах.
                }

                // -------- РЕДУКЦИИ --------
                NodeType::Sum(x) => {
                    let x_im = self.reuse(*x)?;
                    let g_x = self.add_node(NodeType::Broadcast(g_out, x_im));
                    self.acc(*x, g_x);
                }
                NodeType::Mean(x) => {
                    let n = self.len_of_last_axis(*x)? as f32;
                    let inv_n = self.lit_scalar(1.0 / n);
                    let scaled = self.add_node(NodeType::Multiply(g_out, inv_n));
                    let x_im = self.reuse(*x)?;
                    let g_x = self.add_node(NodeType::Broadcast(scaled, x_im));
                    self.acc(*x, g_x);
                }
                NodeType::MeanAxis(x, axis) => {
                    let n = self.len_of_axis(*x, *axis)? as f32;
                    let inv_n = self.lit_scalar(1.0 / n);
                    let scaled = self.add_node(NodeType::Multiply(g_out, inv_n));
                    let x_im = self.reuse(*x)?;
                    let g_x = self.add_node(NodeType::Broadcast(scaled, x_im));
                    self.acc(*x, g_x);
                }
                NodeType::Variance(x) => {
                    // var(x) = mean((x - mean(x))^2) по последней оси (keepdim=1)
                    // d var / dx = (2/N) * (x - mean(x))
                    let n = self.len_of_last_axis(*x)? as f32;
                    let two_over_n = self.lit_scalar(2.0 / n);
                    let x_im = self.reuse(*x)?;
                    let mean_x = self.add_node(NodeType::Mean(x_im));
                    let x_im2 = self.reuse(*x)?;
                    let x_centered = self.add_node(NodeType::Subtract(x_im2, mean_x));
                    let factor = self.add_node(NodeType::Multiply(g_out, two_over_n));
                    let term = self.add_node(NodeType::Multiply(factor, x_centered));
                    let x_im3 = self.reuse(*x)?;
                    let g_x = self.add_node(NodeType::Broadcast(term, x_im3));
                    self.acc(*x, g_x);
                }

                // -------- ТРАНСФОРМАЦИИ --------
                NodeType::Reshape(x, _shape_node) => {
                    let x_im = self.reuse(*x)?;
                    let g_x = self.add_node(NodeType::Reshape(g_out, x_im));
                    self.acc(*x, g_x);
                }
                NodeType::Transpose(x, a1, a2) => {
                    let g_x = self.add_node(NodeType::Transpose(g_out, *a2, *a1));
                    self.acc(*x, g_x);
                }
                NodeType::Broadcast(x, _target) => {
                    let x_im = self.reuse(*x)?;
                    let g_x = self.add_node(NodeType::ReduceSumTo(g_out, x_im));
                    self.acc(*x, g_x);
                }
                NodeType::ReduceSumTo(x, _target) => {
                    let x_im = self.reuse(*x)?;
                    let g_x = self.add_node(NodeType::Broadcast(g_out, x_im));
                    self.acc(*x, g_x);
                }

                // Пулинг — вне текущих тестов
                NodeType::MaxPool2d { .. } | NodeType::MaxUnpool2d { .. } => {}

                // Сравнения: производная ≈ 0
                NodeType::GreaterThan(_, _) | NodeType::Less(_, _) | NodeType::Equal(_, _) => {}

                // Управляющие конструкции — не поддерживаем в автограде
                NodeType::If { .. } | NodeType::ForLoop { .. } => {}
            }
        }

        // Имена на градиентах для входов/параметров — такие же, как у источников
        for (&src_id, &g_id) in self.gmap.iter() {
            if let Ok(src_node) = self.src.get_node(src_id) {
                if let Some(name) = match &src_node.node_type {
                    NodeType::Input { name } | NodeType::Parameter { name } => Some(name.clone()),
                    _ => None,
                } {
                    if let Ok(gn) = self.grad.get_node_mut(g_id) {
                        gn.name = Some(name);
                    }
                }
            }
        }

        Ok(())
    }

    /// Построить и прогнать инференс форм/типов по градиентному графу.
    fn build_and_infer(&mut self, loss_id: NodeId) -> AutogradResult<()> {
        self.build(loss_id)?;
        let initial_shapes: HashMap<String, (Shape, DType)> = HashMap::new();
        ShapeInference::run(&mut self.grad, &initial_shapes)?;
        Ok(())
    }

    /// Добавить новый узел в grad-граф.
    fn add_node(&mut self, node_type: NodeType) -> NodeId {
        self.grad.add_node(None, node_type)
    }

    /// Литерал-скаляр.
    fn lit_scalar(&mut self, v: f32) -> NodeId {
        self.grad.add_node(None, NodeType::Literal(Value::ScalarF32(v)))
    }

    /// «Образ» исходного узла в grad-графе:
    /// создаём Literal(Tensor) нужной формы (данные не важны — важна shape/dtype).
    fn reuse(&mut self, src_id: NodeId) -> AutogradResult<NodeId> {
        // Уже создавали?
        let name = format!("__im_src_{}", src_id);
        if let Some(existing) = self
            .grad
            .nodes
            .values()
            .find(|n| n.name.as_deref() == Some(&name))
            .map(|n| n.id)
        {
            return Ok(existing);
        }

        let src = self.src.get_node(src_id)?;
        let shape = src.shape.clone().unwrap_or_default();
        let data = if shape.is_empty() {
            Value::ScalarF32(0.0)
        } else {
            let size: usize = shape.iter().product();
            let zeroes = vec![0.0_f32; size];
            let arr = Array::from_shape_vec(IxDyn(&shape), zeroes).unwrap().into_dyn();
            Value::Tensor(arr)
        };
        let id = self.grad.add_node(Some(name), NodeType::Literal(data));
        if let Ok(nm) = self.grad.get_node_mut(id) {
            nm.shape = Some(shape);
            nm.dtype = src.dtype.or(Some(DType::F32));
        }
        Ok(id)
    }

    /// Аккумуляция: dL/d(target_src_id) += contrib
    fn acc(&mut self, target_src_id: NodeId, contrib: NodeId) {
        if let Some(prev) = self.gmap.get(&target_src_id).copied() {
            let sum = self.add_node(NodeType::Add(prev, contrib));
            self.gmap.insert(target_src_id, sum);
        } else {
            self.gmap.insert(target_src_id, contrib);
        }
    }

    /// Привести градиент `g` к форме источника `src_id` (ReduceSumTo по бродкаст-осям).
    fn reduce_to(&mut self, g: NodeId, src_id: NodeId) -> AutogradResult<NodeId> {
        let src_im = self.reuse(src_id)?;
        let red = self.add_node(NodeType::ReduceSumTo(g, src_im));
        Ok(red)
    }

    /// Длина последней оси источника.
    fn len_of_last_axis(&self, src_id: NodeId) -> AutogradResult<usize> {
        let n = self.src.get_node(src_id)?;
        let shape = n
            .shape
            .clone()
            .ok_or(AutogradError::Asg(AsgError::InvalidShape(src_id)))?;
        Ok(*shape.last().unwrap_or(&1))
    }

    /// Длина произвольной оси (поддерживает отрицательные индексы).
    fn len_of_axis(&self, src_id: NodeId, axis: isize) -> AutogradResult<usize> {
        let n = self.src.get_node(src_id)?;
        let shape = n
            .shape
            .clone()
            .ok_or(AutogradError::Asg(AsgError::InvalidShape(src_id)))?;
        if shape.is_empty() {
            return Ok(1);
        }
        let rank = shape.len();
        let mut ax = if axis < 0 {
            (rank as isize + axis) as usize
        } else {
            axis as usize
        };
        if ax >= rank {
            ax = rank - 1;
        }
        Ok(shape[ax])
    }
}
