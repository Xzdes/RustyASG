//! Module defining the core of the Abstract Semantic Graph (ASG).
//!
//! ASG is the fundamental representation of any computational task in the
//! `RustyGradients` ecosystem. Instead of immediately executing operations,
//! the framework first builds a graph that describes data and the sequence
//! of operations on them.
//!
//! This approach enables complex graph analysis, optimization,
//! JIT compilation for various backends (CPU, GPU), and automatic
//! differentiation.
//!
//! # Key Components:
//!
//! - `Asg`: The graph itself, a collection of nodes and information about its inputs/outputs.
//! - `Node`: A node in the graph, representing either data or an operation.
//! - `NodeType`: Enumeration of all possible operations (math, logic, I/O).
//! - `Value`: Enumeration of all possible data types that the graph operates on.
//! - `NodeId`, `AsgId`: Unique identifiers for nodes and graphs.
//! - `Shape`, `DType`: Information about tensor shape and data type at the graph level.

use ndarray::ArrayD;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Unique identifier for a node in the graph.
pub type NodeId = usize;

/// Unique identifier for the graph itself (useful for nested graphs).
pub type AsgId = usize;

/// Representation of tensor shape (dimensions).
pub type Shape = Vec<usize>;

/// Result type for ASG-related operations.
pub type AsgResult<T> = std::result::Result<T, AsgError>;

/// Errors that can occur when working with ASG.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum AsgError {
    #[error("Node with ID {0} not found in graph. \
             Verify that the node was added to the graph using add_node() before use.")]
    NodeNotFound(NodeId),

    #[error("Input with name '{0}' not found in graph. \
             Ensure that an Input node was created with this name.")]
    InputNotFound(String),

    #[error("Invalid graph structure: {0}")]
    InvalidGraph(String),

    #[error("Cyclic dependency detected in graph. \
             ASG must be a directed acyclic graph (DAG).")]
    CyclicDependency,
}

/// Enumeration of basic data types supported at the graph level.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DType {
    F32,
    I64,
    Bool,
}

/// Enumeration of all possible data types (values) that can exist
/// in the graph during execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Value {
    /// Standard multidimensional array (tensor) for numerical computations.
    Tensor(ArrayD<f32>),
    /// 64-bit integer.
    Integer(i64),
    /// 32-bit floating point number.
    Float(f32),
    /// Boolean value.
    Boolean(bool),
    /// Text string.
    Text(String),
    /// Unit type, analogous to `()` in Rust. Used for operations that
    /// don't return a meaningful result (e.g., `Print`).
    Unit,
}

/// A node in the Abstract Semantic Graph.
///
/// Each node has a unique ID and a type that determines whether the node is
/// data (e.g., `Literal`) or an operation (`MatrixMultiply`).
/// Also stores meta-information about shape and data type, computed
/// during Shape Inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    /// Unique ID of the node within its graph.
    pub id: NodeId,
    /// Optional name for debugging and visualization.
    pub name: Option<String>,
    /// Node type that determines its behavior.
    pub node_type: NodeType,
    /// Shape of the tensor produced by this node. `None` if not yet computed.
    pub shape: Option<Shape>,
    /// Data type of the tensor. `None` if not yet computed.
    pub dtype: Option<DType>,
}

/// Enumeration of all possible operations and data types in the graph.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum NodeType {
    // --- Data and Input Nodes ---
    /// Input node of the graph. Defines the public API of the graph.
    Input { name: String },
    /// Trainable model parameter (e.g., weights or biases).
    Parameter { name: String },
    /// Constant value (literal) embedded directly in the graph.
    Literal(Value),
    /// Reference to a node in another "external" graph.
    /// Used in gradient graph to reference values from the forward pass graph.
    External {
        name: String,
        source_asg_id: AsgId,
        source_node_id: NodeId,
    },

    // --- Mathematical and Logical Operations ---
    Add(NodeId, NodeId),
    Subtract(NodeId, NodeId),
    Multiply(NodeId, NodeId),
    Divide(NodeId, NodeId),
    MatrixMultiply(NodeId, NodeId),
    GreaterThan(NodeId, NodeId), // Element-wise comparison >

    // --- Element-wise Operations ---
    ReLU(NodeId),
    Sigmoid(NodeId),
    Log(NodeId),
    Sqrt(NodeId),
    Exp(NodeId),
    Abs(NodeId),
    Neg(NodeId),
    Power(NodeId, NodeId), // Second argument is the exponent (can be a constant)
    Softmax(NodeId),       // Softmax along the last axis

    // --- Additional Activations ---
    Tanh(NodeId),
    LeakyReLU(NodeId, f32), // (input, negative_slope)
    GELU(NodeId),
    SiLU(NodeId), // also known as Swish
    ELU(NodeId, f32), // (input, alpha)
    Softplus(NodeId, f32), // (input, beta)
    Clamp(NodeId, f32, f32), // (input, min, max)

    // --- Reduction Operations ---
    Sum(NodeId),      // Sum of all tensor elements
    Mean(NodeId),     // Mean along the last axis
    Variance(NodeId), // Variance along the last axis

    // --- Embedding Operations ---
    /// Embedding lookup: transforms indices into dense vectors.
    /// indices: [*] (any shape), weight: [num_embeddings, embedding_dim]
    /// Output: [*, embedding_dim]
    Embedding {
        /// Index tensor (integers represented as f32).
        indices: NodeId,
        /// Embedding matrix of shape [num_embeddings, embedding_dim].
        weight: NodeId,
    },
    /// Gradient for Embedding: scatter-add operation.
    /// Accumulates gradients into the weight matrix by indices.
    /// grad_output: [*, embedding_dim], indices: [*]
    /// Output: [num_embeddings, embedding_dim]
    EmbeddingGrad {
        /// Gradient from subsequent operations [*, embedding_dim].
        grad_output: NodeId,
        /// Indices used for the lookup.
        indices: NodeId,
        /// Number of embeddings (vocabulary size).
        num_embeddings: usize,
    },

    // --- Transformation Operations ---
    Reshape(NodeId, NodeId), // Second argument is tensor with new shape
    Transpose(NodeId, usize, usize), // Axes to transpose
    /// Broadcasts the first tensor (scalar) to the shape of the second tensor.
    /// Used mainly in gradient graph, e.g., for grad(Sum).
    Broadcast(NodeId, NodeId),
    /// Sums tensor `source` so that its shape matches the shape of `target_shape_provider`.
    /// Used in autograd for gradients from broadcast operations.
    ReduceSumTo(NodeId, NodeId),

    // --- Convolution Operations ---
    /// 2D Convolution.
    /// Input tensor of shape [N, C_in, H, W], kernel of shape [C_out, C_in, kH, kW].
    Conv2d {
        /// Input tensor of shape [N, C_in, H, W].
        input: NodeId,
        /// Convolution kernel (weights) of shape [C_out, C_in/groups, kH, kW].
        weight: NodeId,
        /// Optional bias of shape [C_out].
        bias: Option<NodeId>,
        /// Stride (stride_h, stride_w).
        stride: (usize, usize),
        /// Padding (pad_h, pad_w).
        padding: (usize, usize),
        /// Dilation (dilation_h, dilation_w).
        dilation: (usize, usize),
        /// Number of groups for grouped convolution.
        groups: usize,
    },

    /// Transposed 2D Convolution (Deconvolution).
    ConvTranspose2d {
        input: NodeId,
        weight: NodeId,
        bias: Option<NodeId>,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    },

    // --- Pooling Operations ---
    MaxPool2d {
        /// Input tensor, typically of shape [N, C, H, W].
        input: NodeId,
        /// Window size, e.g., (2, 2).
        kernel_size: (usize, usize),
        /// Window stride, e.g., (2, 2).
        stride: (usize, usize),
    },

    MaxUnpool2d {
        /// Input gradient (from upstream layer).
        input: NodeId,
        /// Reference to the ORIGINAL MaxPool2d input (needed to determine output shape).
        original_input: NodeId,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    },

    /// Average Pooling 2D.
    AvgPool2d {
        input: NodeId,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    },

    /// Backward pass for AvgPool2d.
    /// Distributes gradient uniformly across the pooling window.
    AvgUnpool2d {
        /// Gradient from next layer.
        input: NodeId,
        /// Reference to original AvgPool2d input for shape.
        original_input: NodeId,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    },

    /// Adaptive Average Pooling 2D - automatically computes parameters
    /// to achieve the desired output size.
    AdaptiveAvgPool2d {
        input: NodeId,
        /// Target output size (H_out, W_out).
        output_size: (usize, usize),
    },

    /// Layer Normalization - normalizes across the last normalized_shape dimensions.
    /// y = gamma * (x - mean) / sqrt(var + eps) + beta
    LayerNorm {
        /// Input tensor.
        input: NodeId,
        /// Trainable scale parameter.
        gamma: NodeId,
        /// Trainable shift parameter.
        beta: NodeId,
        /// Small constant for numerical stability.
        eps: f32,
    },

    /// Backward pass for LayerNorm with respect to input x.
    /// Computes the correct gradient considering all dependencies through mean and variance.
    LayerNormBackward {
        /// Gradient from the next layer ∂L/∂y.
        grad_output: NodeId,
        /// Original input x.
        input: NodeId,
        /// Gamma parameter.
        gamma: NodeId,
        /// Epsilon for numerical stability.
        eps: f32,
    },

    /// Gradient of LayerNorm with respect to parameter gamma.
    /// grad_gamma = sum(grad_output * x_normalized, axis=batch)
    LayerNormGradGamma {
        /// Gradient from the next layer ∂L/∂y.
        grad_output: NodeId,
        /// Original input x.
        input: NodeId,
        /// Epsilon for numerical stability.
        eps: f32,
    },

    /// Gradient of LayerNorm with respect to parameter beta.
    /// grad_beta = sum(grad_output, axis=batch)
    LayerNormGradBeta {
        /// Gradient from the next layer ∂L/∂y.
        grad_output: NodeId,
    },

    /// Gradient of Conv2d with respect to input.
    /// Implemented as transposed convolution.
    Conv2dBackwardInput {
        /// Gradient from next layer (d_output) of shape [N, C_out, H_out, W_out].
        grad_output: NodeId,
        /// Convolution weights of shape [C_out, C_in/groups, kH, kW].
        weight: NodeId,
        /// Original input shape [N, C_in, H_in, W_in].
        input_shape: (usize, usize, usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    },

    /// Gradient of Conv2d with respect to weights.
    /// Implemented as convolution of input with output gradient.
    Conv2dBackwardWeight {
        /// Gradient from next layer of shape [N, C_out, H_out, W_out].
        grad_output: NodeId,
        /// Original input of shape [N, C_in, H_in, W_in].
        input: NodeId,
        /// Weight shape [C_out, C_in/groups, kH, kW].
        weight_shape: (usize, usize, usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        groups: usize,
    },

    // --- Control Flow Constructs ---
    /// Conditional execution. Executes one of two sub-graphs depending
    /// on the condition.
    If {
        condition: NodeId,
        then_asg: AsgId,
        else_asg: AsgId,
    },
    /// Loop execution of a sub-graph.
    ForLoop {
        iterable: NodeId,      // Node to iterate over (e.g., tensor)
        loop_body_asg: AsgId,  // ID of sub-graph that will be the loop body
    },

    // --- Functions ---
    /// Function definition within the graph.
    FunctionDefinition {
        name: String,
        body_asg: AsgId,
    },
    /// Call to a previously defined function.
    FunctionCall {
        function_id: NodeId, // ID of FunctionDefinition node
        args: Vec<NodeId>,
    },

    // --- I/O and Side Effects ---
    /// Prints the value of a node to standard output during execution.
    Print(NodeId),
}

/// Structure representing the Abstract Semantic Graph itself.
///
/// Contains a collection of all nodes and defines the graph's "interface" —
/// its inputs and output nodes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Asg {
    /// Unique ID of the graph.
    pub id: AsgId,
    /// Optional name for debugging.
    pub name: Option<String>,
    /// All nodes belonging to this graph, stored by their ID.
    pub nodes: HashMap<NodeId, Node>,
    /// IDs of nodes that are inputs to this graph.
    pub inputs: Vec<NodeId>,
    /// IDs of nodes that are the result of computing the entire graph.
    pub outputs: Vec<NodeId>,
}

impl Asg {
    /// Creates a new, empty graph with the given ID.
    pub fn new(id: AsgId, name: Option<String>) -> Self {
        Self {
            id,
            name,
            nodes: HashMap::new(),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Adds a new node to the graph.
    /// At this stage, shape and data type information is absent.
    pub fn add_node(&mut self, name: Option<String>, node_type: NodeType) -> NodeId {
        let new_id = self.nodes.len();
        let node = Node {
            id: new_id,
            name,
            node_type,
            shape: None,
            dtype: None,
        };
        self.nodes.insert(new_id, node);
        new_id
    }

    /// Sets the input nodes for the graph.
    pub fn set_inputs(&mut self, inputs: Vec<NodeId>) {
        self.inputs = inputs;
    }

    /// Sets the output nodes for the graph.
    pub fn set_outputs(&mut self, outputs: Vec<NodeId>) {
        self.outputs = outputs;
    }

    /// Helper method for setting a single output node.
    pub fn set_output(&mut self, output: NodeId) {
        self.outputs = vec![output];
    }

    /// Finds a node by its ID.
    pub fn get_node(&self, id: NodeId) -> AsgResult<&Node> {
        self.nodes
            .get(&id)
            .ok_or(AsgError::NodeNotFound(id))
    }

    /// Finds a mutable node by its ID.
    pub fn get_node_mut(&mut self, id: NodeId) -> AsgResult<&mut Node> {
        self.nodes
            .get_mut(&id)
            .ok_or(AsgError::NodeNotFound(id))
    }
}
