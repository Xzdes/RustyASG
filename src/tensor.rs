//! Tensor API и GraphContext

use crate::asg::{Asg, NodeId, NodeType, Value};
use ndarray::{arr1, Array1, ArrayD, IxDyn};
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

#[derive(Debug)]
pub struct GraphContext {
    main_graph: Asg,
}

impl GraphContext {
    pub fn new() -> Self {
        Self { main_graph: Asg::new(0, Some("main".to_string())) }
    }
    pub fn main_graph(&self) -> &Asg { &self.main_graph }
    pub fn main_graph_mut(&mut self) -> &mut Asg { &mut self.main_graph }
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub node_id: NodeId,
    pub context: Rc<RefCell<GraphContext>>,
}

impl Tensor {
    pub fn new_input(context: &Rc<RefCell<GraphContext>>, name: &str) -> Self {
        let node_id = context.borrow_mut().main_graph_mut().add_node(
            Some(name.to_string()),
            NodeType::Input { name: name.to_string() },
        );
        context.borrow_mut().main_graph_mut().inputs.push(node_id);
        Self { node_id, context: Rc::clone(context) }
    }

    pub fn new_parameter(context: &Rc<RefCell<GraphContext>>, name: &str) -> Self {
        let node_id = context.borrow_mut().main_graph_mut().add_node(
            Some(name.to_string()),
            NodeType::Parameter { name: name.to_string() },
        );
        Self { node_id, context: Rc::clone(context) }
    }

    pub fn new_literal(context: &Rc<RefCell<GraphContext>>, data: ArrayD<f32>, name: &str) -> Self {
        let node_id = context.borrow_mut().main_graph_mut().add_node(
            Some(name.to_string()),
            NodeType::Literal(Value::Tensor(data)),
        );
        Self { node_id, context: Rc::clone(context) }
    }

    pub fn dot(&self, rhs: &Tensor) -> Self {
        let id = self.context.borrow_mut().main_graph_mut().add_node(None, NodeType::MatrixMultiply(self.node_id, rhs.node_id));
        Self { node_id: id, context: Rc::clone(&self.context) }
    }

    /// Ожидаем вызовы вида: `x.reshape(vec![...])`
    pub fn reshape(&self, shape: Vec<i64>) -> Self {
        // готовим литерал формы как ArrayD<f32> с динамической размерностью [len]
        let shape_f32: Vec<f32> = shape.into_iter().map(|v| v as f32).collect();
        let arr: Array1<f32> = arr1(&shape_f32);
        let arr_dyn: ArrayD<f32> = ArrayD::from_shape_vec(IxDyn(&[arr.len()]), arr.to_vec()).unwrap();

        let shape_node = self.context.borrow_mut().main_graph_mut().add_node(
            Some("shape".to_string()),
            NodeType::Literal(Value::Tensor(arr_dyn)),
        );
        let id = self.context.borrow_mut().main_graph_mut().add_node(None, NodeType::Reshape(self.node_id, shape_node));
        Self { node_id: id, context: Rc::clone(&self.context) }
    }

    pub fn transpose(&self, axis1: usize, axis2: usize) -> Self {
        let id = self.context.borrow_mut().main_graph_mut().add_node(None, NodeType::Transpose(self.node_id, axis1, axis2));
        Self { node_id: id, context: Rc::clone(&self.context) }
    }

    pub fn broadcast_to(&self, shape_provider: &Tensor) -> Self {
        let id = self.context.borrow_mut().main_graph_mut().add_node(None, NodeType::Broadcast(self.node_id, shape_provider.node_id));
        Self { node_id: id, context: Rc::clone(&self.context) }
    }

    pub fn exp(&self) -> Self {
        let id = self.context.borrow_mut().main_graph_mut().add_node(None, NodeType::Exp(self.node_id));
        Self { node_id: id, context: Rc::clone(&self.context) }
    }

    pub fn log(&self) -> Self {
        let id = self.context.borrow_mut().main_graph_mut().add_node(None, NodeType::Log(self.node_id));
        Self { node_id: id, context: Rc::clone(&self.context) }
    }

    pub fn sqrt(&self) -> Self {
        let id = self.context.borrow_mut().main_graph_mut().add_node(None, NodeType::Sqrt(self.node_id));
        Self { node_id: id, context: Rc::clone(&self.context) }
    }

    pub fn relu(&self) -> Self {
        let id = self.context.borrow_mut().main_graph_mut().add_node(None, NodeType::ReLU(self.node_id));
        Self { node_id: id, context: Rc::clone(&self.context) }
    }

    pub fn sigmoid(&self) -> Self {
        let id = self.context.borrow_mut().main_graph_mut().add_node(None, NodeType::Sigmoid(self.node_id));
        Self { node_id: id, context: Rc::clone(&self.context) }
    }

    pub fn softmax(&self) -> Self {
        let id = self.context.borrow_mut().main_graph_mut().add_node(None, NodeType::Softmax(self.node_id));
        Self { node_id: id, context: Rc::clone(&self.context) }
    }

    pub fn sum(&self) -> Self {
        let id = self.context.borrow_mut().main_graph_mut().add_node(None, NodeType::Sum(self.node_id));
        Self { node_id: id, context: Rc::clone(&self.context) }
    }

    pub fn mean(&self) -> Self {
        let id = self.context.borrow_mut().main_graph_mut().add_node(None, NodeType::Mean(self.node_id));
        Self { node_id: id, context: Rc::clone(&self.context) }
    }

    pub fn mean_axis(&self, axis: isize) -> Self {
        let id = self.context.borrow_mut().main_graph_mut().add_node(None, NodeType::MeanAxis(self.node_id, axis));
        Self { node_id: id, context: Rc::clone(&self.context) }
    }

    pub fn variance(&self) -> Self {
        let id = self.context.borrow_mut().main_graph_mut().add_node(None, NodeType::Variance(self.node_id));
        Self { node_id: id, context: Rc::clone(&self.context) }
    }
}

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        let id = self.context.borrow_mut().main_graph_mut().add_node(None, NodeType::Add(self.node_id, rhs.node_id));
        Tensor { node_id: id, context: Rc::clone(&self.context) }
    }
}
impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        let id = self.context.borrow_mut().main_graph_mut().add_node(None, NodeType::Subtract(self.node_id, rhs.node_id));
        Tensor { node_id: id, context: Rc::clone(&self.context) }
    }
}
impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Self::Output {
        let id = self.context.borrow_mut().main_graph_mut().add_node(None, NodeType::Multiply(self.node_id, rhs.node_id));
        Tensor { node_id: id, context: Rc::clone(&self.context) }
    }
}
impl Div<&Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Self::Output {
        let id = self.context.borrow_mut().main_graph_mut().add_node(None, NodeType::Divide(self.node_id, rhs.node_id));
        Tensor { node_id: id, context: Rc::clone(&self.context) }
    }
}

// удобные комбинации владений
impl Add<Tensor> for &Tensor { type Output = Tensor; fn add(self, rhs: Tensor) -> Self::Output { self.add(&rhs) } }
impl Add<&Tensor> for Tensor { type Output = Tensor; fn add(self, rhs: &Tensor) -> Self::Output { (&self).add(rhs) } }
impl Add<Tensor> for Tensor { type Output = Tensor; fn add(self, rhs: Tensor) -> Self::Output { (&self).add(&rhs) } }

impl Sub<Tensor> for &Tensor { type Output = Tensor; fn sub(self, rhs: Tensor) -> Self::Output { self.sub(&rhs) } }
impl Sub<&Tensor> for Tensor { type Output = Tensor; fn sub(self, rhs: &Tensor) -> Self::Output { (&self).sub(rhs) } }
impl Sub<Tensor> for Tensor { type Output = Tensor; fn sub(self, rhs: Tensor) -> Self::Output { (&self).sub(&rhs) } }

impl Mul<Tensor> for &Tensor { type Output = Tensor; fn mul(self, rhs: Tensor) -> Self::Output { self.mul(&rhs) } }
impl Mul<&Tensor> for Tensor { type Output = Tensor; fn mul(self, rhs: &Tensor) -> Self::Output { (&self).mul(rhs) } }
impl Mul<Tensor> for Tensor { type Output = Tensor; fn mul(self, rhs: Tensor) -> Self::Output { (&self).mul(&rhs) } }

impl Div<Tensor> for &Tensor { type Output = Tensor; fn div(self, rhs: Tensor) -> Self::Output { self.div(&rhs) } }
impl Div<&Tensor> for Tensor { type Output = Tensor; fn div(self, rhs: &Tensor) -> Self::Output { (&self).div(rhs) } }
impl Div<Tensor> for Tensor { type Output = Tensor; fn div(self, rhs: Tensor) -> Self::Output { (&self).div(&rhs) } }
