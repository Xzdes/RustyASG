//! Module defining `Tensor` and `GraphContext`.
//!
//! In the new architecture, `Tensor` is no longer a data container.
//! Instead, it is a lightweight "handle" or "symbolic variable"
//! that represents a node in the `Abstract Semantic Graph` (ASG).
//!
//! All tensor operations (`add`, `dot`, etc.) do not perform computations
//! immediately, but add corresponding nodes to the graph.
//!
//! `GraphContext` is the central object that owns and manages
//! ASG construction.

use crate::asg::{Asg, NodeId, NodeType, Value};
use ndarray::ArrayD;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

/// Context that owns and manages construction of one or more ASGs.
///
/// This object is wrapped in `Rc<RefCell<>>` so it can be safely
/// shared between multiple `Tensor` handles.
#[derive(Debug, Clone)]
pub struct GraphContext {
    // For simplicity, we currently work with a single main graph.
    // In the future, this may be a collection of graphs to support nesting.
    main_graph: Asg,
}

impl GraphContext {
    /// Creates a new, empty graph context.
    pub fn new() -> Self {
        Self {
            main_graph: Asg::new(0, Some("main".to_string())),
        }
    }

    /// Gets a mutable reference to the main graph for construction.
    pub fn main_graph_mut(&mut self) -> &mut Asg {
        &mut self.main_graph
    }

    /// Gets an immutable reference to the main graph.
    pub fn main_graph(&self) -> &Asg {
        &self.main_graph
    }
}

impl Default for GraphContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Symbolic handle representing a node in the computation graph (ASG).
///
/// This object does not contain actual data. It consists of a node ID and a reference
/// to the `GraphContext` where this node exists.
///
/// Any operation on this object results in adding a new node to the graph.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Node ID in the ASG that this tensor represents.
    pub node_id: NodeId,
    /// Shared reference to the context where the graph is being built.
    pub context: Rc<RefCell<GraphContext>>,
}

impl Tensor {
    /// Creates a new "input" node in the graph and returns a handle for it.
    /// Input nodes are "variables" of the graph into which actual data
    /// will be fed during execution.
    pub fn new_input(context: &Rc<RefCell<GraphContext>>, name: &str) -> Self {
        let mut ctx = context.borrow_mut();
        let graph = ctx.main_graph_mut();

        let node_id = graph.add_node(
            Some(name.to_string()),
            NodeType::Input {
                name: name.to_string(),
            },
        );

        // Register this node as one of the graph's inputs.
        graph.inputs.push(node_id);

        Self {
            node_id,
            context: Rc::clone(context),
        }
    }

    /// Creates a new "parameter" in the graph.
    /// Parameters are trainable model weights.
    pub fn new_parameter(context: &Rc<RefCell<GraphContext>>, name: &str) -> Self {
        let node_id = context.borrow_mut().main_graph_mut().add_node(
            Some(name.to_string()),
            NodeType::Parameter {
                name: name.to_string(),
            },
        );
        Self {
            node_id,
            context: Rc::clone(context),
        }
    }

    /// Creates a new constant node (literal) from actual data.
    /// This data will be embedded directly into the graph.
    pub fn new_literal(context: &Rc<RefCell<GraphContext>>, data: ArrayD<f32>, name: &str) -> Self {
        let node_id = context.borrow_mut().main_graph_mut().add_node(
            Some(name.to_string()),
            NodeType::Literal(Value::Tensor(data)),
        );
        Self {
            node_id,
            context: Rc::clone(context),
        }
    }

    // --- Mathematical operations ---

    pub fn pow(&self, power: &Tensor) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Power(self.node_id, power.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    pub fn dot(&self, other: &Tensor) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::MatrixMultiply(self.node_id, other.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    /// Alias for `dot` - matrix multiplication
    pub fn matmul(&self, other: &Tensor) -> Self {
        self.dot(other)
    }

    /// Raise to a scalar power (creates a literal for the exponent)
    pub fn pow_scalar(&self, power: f32) -> Self {
        let exp = Tensor::new_literal(
            &self.context,
            ndarray::arr0(power).into_dyn(),
            "pow_exp",
        );
        self.pow(&exp)
    }

    pub fn sqrt(&self) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Sqrt(self.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    // --- Activation functions ---

    pub fn relu(&self) -> Self {
        let node_id = self
            .context
            .borrow_mut()
            .main_graph_mut()
            .add_node(None, NodeType::ReLU(self.node_id));
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    pub fn sigmoid(&self) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Sigmoid(self.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }
    
    pub fn softmax(&self) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Softmax(self.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    // --- Additional activation functions ---

    /// Tanh activation: (e^x - e^-x) / (e^x + e^-x)
    pub fn tanh(&self) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Tanh(self.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    /// LeakyReLU activation: max(0, x) + negative_slope * min(0, x)
    pub fn leaky_relu(&self, negative_slope: f32) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::LeakyReLU(self.node_id, negative_slope),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    /// GELU activation: x * Φ(x) where Φ is the CDF of the standard normal distribution
    pub fn gelu(&self) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::GELU(self.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    /// SiLU/Swish activation: x * sigmoid(x)
    pub fn silu(&self) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::SiLU(self.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    /// ELU activation: x if x > 0 else alpha * (e^x - 1)
    pub fn elu(&self, alpha: f32) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::ELU(self.node_id, alpha),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    /// Softplus activation: log(1 + e^(beta*x)) / beta
    pub fn softplus(&self, beta: f32) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Softplus(self.node_id, beta),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    /// Exponential: e^x
    pub fn exp(&self) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Exp(self.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    /// Absolute value: |x|
    pub fn abs(&self) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Abs(self.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    /// Negation: -x
    pub fn neg(&self) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Neg(self.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    /// Clamp values to [min, max]
    pub fn clamp(&self, min_val: f32, max_val: f32) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Clamp(self.node_id, min_val, max_val),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    /// Natural logarithm: ln(x)
    pub fn log(&self) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Log(self.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    /// Square: x^2
    pub fn square(&self) -> Self {
        let two = Tensor::scalar(&self.context, 2.0);
        self.pow(&two)
    }

    /// Creates a scalar tensor (0-dimensional) with the given value.
    /// This is useful for creating constants in loss functions.
    pub fn scalar(context: &Rc<RefCell<GraphContext>>, value: f32) -> Self {
        let data = ArrayD::from_elem(ndarray::IxDyn(&[]), value);
        let node_id = context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Literal(Value::Tensor(data)),
        );
        Self {
            node_id,
            context: Rc::clone(context),
        }
    }

    /// Creates a 1D tensor from a vector of values.
    pub fn from_vec(context: &Rc<RefCell<GraphContext>>, values: Vec<f32>) -> Self {
        let len = values.len();
        let data = ArrayD::from_shape_vec(ndarray::IxDyn(&[len]), values).unwrap();
        let node_id = context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Literal(Value::Tensor(data)),
        );
        Self {
            node_id,
            context: Rc::clone(context),
        }
    }

    // --- Reduction operations ---

    pub fn sum(&self) -> Self {
        let node_id = self
            .context
            .borrow_mut()
            .main_graph_mut()
            .add_node(None, NodeType::Sum(self.node_id));
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    pub fn mean(&self) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Mean(self.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    pub fn variance(&self) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Variance(self.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }
    
    // --- Transformation operations ---

    pub fn reshape(&self, shape: Vec<i64>) -> Self {
        // Create a literal node containing the new shape
        let shape_data_f32: Vec<f32> = shape.iter().map(|&x| x as f32).collect();
        let shape_array = ArrayD::from_shape_vec(ndarray::IxDyn(&[shape.len()]), shape_data_f32).unwrap();
        
        let shape_node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Literal(Value::Tensor(shape_array)),
        );

        let reshape_node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Reshape(self.node_id, shape_node_id),
        );
        Self {
            node_id: reshape_node_id,
            context: Rc::clone(&self.context),
        }
    }

    pub fn transpose(&self, axis1: usize, axis2: usize) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Transpose(self.node_id, axis1, axis2),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    pub fn max_pool2d(&self, kernel_size: (usize, usize), stride: (usize, usize)) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::MaxPool2d {
                input: self.node_id,
                kernel_size,
                stride,
            },
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    /// 2D Convolution operation.
    ///
    /// # Arguments
    /// * `weight` - Convolution kernel of shape [C_out, C_in/groups, kH, kW]
    /// * `bias` - Optional bias of shape [C_out]
    /// * `stride` - Stride (stride_h, stride_w)
    /// * `padding` - Padding (pad_h, pad_w)
    /// * `dilation` - Dilation (dilation_h, dilation_w)
    /// * `groups` - Number of groups for grouped convolution
    pub fn conv2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Conv2d {
                input: self.node_id,
                weight: weight.node_id,
                bias: bias.map(|b| b.node_id),
                stride,
                padding,
                dilation,
                groups,
            },
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    /// Transposed 2D Convolution (Deconvolution).
    pub fn conv_transpose2d(
        &self,
        weight: &Tensor,
        bias: Option<&Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    ) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::ConvTranspose2d {
                input: self.node_id,
                weight: weight.node_id,
                bias: bias.map(|b| b.node_id),
                stride,
                padding,
                output_padding,
                dilation,
                groups,
            },
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    /// Average Pooling 2D.
    pub fn avg_pool2d(
        &self,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::AvgPool2d {
                input: self.node_id,
                kernel_size,
                stride,
                padding,
            },
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    /// Adaptive Average Pooling 2D.
    /// Automatically calculates parameters to achieve target output size.
    pub fn adaptive_avg_pool2d(&self, output_size: (usize, usize)) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::AdaptiveAvgPool2d {
                input: self.node_id,
                output_size,
            },
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    /// Embedding lookup: converts indices to dense vectors.
    ///
    /// # Arguments
    /// * `weight` - Embedding matrix of shape [num_embeddings, embedding_dim]
    ///
    /// # Returns
    /// Tensor of shape [*, embedding_dim] where * is the shape of self (indices)
    ///
    /// # Example
    /// ```ignore
    /// let indices = Tensor::new_input(&context, "indices"); // shape: [batch, seq_len]
    /// let embedding_weights = Tensor::new_parameter(&context, "embedding"); // shape: [vocab_size, embed_dim]
    /// let embedded = indices.embedding(&embedding_weights); // shape: [batch, seq_len, embed_dim]
    /// ```
    pub fn embedding(&self, weight: &Tensor) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Embedding {
                indices: self.node_id,
                weight: weight.node_id,
            },
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

}

// Operator implementations for convenient syntax like `a + b`.

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Add(self.node_id, rhs.node_id),
        );
        Tensor {
            node_id,
            context: Rc::clone(&self.context),
        }
    }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Subtract(self.node_id, rhs.node_id),
        );
        Tensor {
            node_id,
            context: Rc::clone(&self.context),
        }
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Self::Output {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Multiply(self.node_id, rhs.node_id),
        );
        Tensor {
            node_id,
            context: Rc::clone(&self.context),
        }
    }
}

impl Div<&Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Self::Output {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Divide(self.node_id, rhs.node_id),
        );
        Tensor {
            node_id,
            context: Rc::clone(&self.context),
        }
    }
}

// --- Add ---
impl Add<Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        self.add(&rhs)
    }
}
impl Add<&Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        (&self).add(rhs)
    }
}
impl Add<Tensor> for Tensor {
    type Output = Tensor;
    fn add(self, rhs: Tensor) -> Self::Output {
        (&self).add(&rhs)
    }
}

// --- Subtract ---
impl Sub<Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        self.sub(&rhs)
    }
}
impl Sub<&Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        (&self).sub(rhs)
    }
}
impl Sub<Tensor> for Tensor {
    type Output = Tensor;
    fn sub(self, rhs: Tensor) -> Self::Output {
        (&self).sub(&rhs)
    }
}

// --- Multiply ---
impl Mul<Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Self::Output {
        self.mul(&rhs)
    }
}
impl Mul<&Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Self::Output {
        (&self).mul(rhs)
    }
}
impl Mul<Tensor> for Tensor {
    type Output = Tensor;
    fn mul(self, rhs: Tensor) -> Self::Output {
        (&self).mul(&rhs)
    }
}

// --- Divide ---
impl Div<Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Self::Output {
        self.div(&rhs)
    }
}
impl Div<&Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Self::Output {
        (&self).div(rhs)
    }
}
impl Div<Tensor> for Tensor {
    type Output = Tensor;
    fn div(self, rhs: Tensor) -> Self::Output {
        (&self).div(&rhs)
    }
}