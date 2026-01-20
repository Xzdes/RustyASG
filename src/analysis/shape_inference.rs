//! Module for shape and data type inference (Shape Inference).
//!
//! Traverses the computation graph and determines the shape
//! and data type of the output tensor for each node based on its input shapes and operation type.

use crate::asg::{Asg, AsgError, DType, Node, NodeId, NodeType, Shape, Value};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum ShapeInferenceError {
    #[error("Graph error: {0}")]
    AsgError(#[from] AsgError),

    #[error("Incompatible shapes for operation '{op}': left operand {shape1:?}, right operand {shape2:?}. \
             Ensure dimensions are compatible for broadcasting or matrix multiplication.")]
    IncompatibleShapes {
        op: String,
        shape1: Shape,
        shape2: Shape,
    },

    #[error("Shape information missing for node {0}. \
             This may mean the node has not been processed by shape inference yet or the graph contains a cyclic dependency.")]
    MissingShapeInfo(NodeId),

    #[error("Initial shape not specified for '{0}'. \
             Add the shape to the initial_shapes HashMap when calling ShapeInference::run().")]
    MissingInitialShape(String),

    #[error("Invalid tensor rank for node {node_id}: expected {expected}D, got {actual}D. \
             Check input data dimensions.")]
    InvalidRank {
        node_id: NodeId,
        expected: usize,
        actual: usize,
    },

    #[error("Node {0} must be a Literal for shape computation (e.g., for Reshape operation). \
             Dynamic shapes are not supported.")]
    NotALiteral(NodeId),

    #[error("Shape inference not implemented for operation: {0}. \
             Add handling for this node type in ShapeInference::infer_node_shape().")]
    UnimplementedNodeType(String),

    #[error("Broadcast error: cannot broadcast shapes {0:?} and {1:?} to a common shape.")]
    BroadcastError(Shape, Shape),

    #[error("Matrix multiplication error: incompatible inner dimensions {0} and {1}.")]
    MatmulDimensionError(usize, usize),
}

type Result<T> = std::result::Result<T, ShapeInferenceError>;

/// Structure that performs shape inference for ASG.
pub struct ShapeInference;

impl ShapeInference {
    /// Runs the shape inference process for the graph.
    ///
    /// Modifies the graph in-place, filling in the `shape` and `dtype` fields for each node.
    ///
    /// # Arguments
    /// * `asg` - Mutable reference to the graph to analyze.
    /// * `initial_shapes` - HashMap providing shapes and types for all
    ///   `Input` and `Parameter` nodes. Key is the node name.
    pub fn run(asg: &mut Asg, initial_shapes: &HashMap<String, (Shape, DType)>) -> Result<()> {
        let sorted_nodes = Self::topological_sort(asg)?;

        for node_id in sorted_nodes {
            let mut node = asg.get_node(node_id)?.clone();

            let (shape, dtype) = Self::infer_node_shape(asg, &node, initial_shapes)?;

            node.shape = Some(shape);
            node.dtype = Some(dtype);
            asg.nodes.insert(node_id, node);
        }

        Ok(())
    }

    /// Main shape inference logic for a single node.
    fn infer_node_shape(
        asg: &Asg,
        node: &Node,
        initial_shapes: &HashMap<String, (Shape, DType)>,
    ) -> Result<(Shape, DType)> {
        match &node.node_type {
            NodeType::Input { name }
            | NodeType::Parameter { name }
            | NodeType::External { name, .. } => initial_shapes
                .get(name)
                .cloned()
                .ok_or_else(|| ShapeInferenceError::MissingInitialShape(name.clone())),

            NodeType::Literal(value) => match value {
                Value::Tensor(arr) => Ok((arr.shape().to_vec(), DType::F32)),
                Value::Integer(_) => Ok((vec![], DType::I64)),
                Value::Boolean(_) => Ok((vec![], DType::Bool)),
                _ => Ok((vec![], DType::F32)),
            },

            NodeType::Add(l, r)
            | NodeType::Subtract(l, r)
            | NodeType::Multiply(l, r)
            | NodeType::Divide(l, r) => {
                let (ls, ld) = Self::get_shape_dtype(asg, *l)?;
                let (rs, _rd) = Self::get_shape_dtype(asg, *r)?;

                let out_shape = if ls.iter().product::<usize>() >= rs.iter().product::<usize>() {
                    ls
                } else {
                    rs
                };

                Ok((out_shape, ld))
            }

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

                let mut out_shape = if ls.len() > 2 { ls[..ls.len() - 2].to_vec() } else { vec![] };
                out_shape.push(m);
                out_shape.push(n);

                Ok((out_shape, ld))
            }

            // Element-wise operations - shape unchanged
            NodeType::ReLU(id) | NodeType::Sigmoid(id) | NodeType::Sqrt(id) | NodeType::Log(id) |
            NodeType::Exp(id) | NodeType::Abs(id) | NodeType::Neg(id) | NodeType::Tanh(id) |
            NodeType::GELU(id) | NodeType::SiLU(id) => {
                Self::get_shape_dtype(asg, *id)
            }

            // Element-wise with parameters - shape unchanged
            NodeType::LeakyReLU(id, _) | NodeType::ELU(id, _) | NodeType::Softplus(id, _) |
            NodeType::Clamp(id, _, _) => {
                Self::get_shape_dtype(asg, *id)
            }

            NodeType::Sum(_) => Ok((vec![], DType::F32)),

            NodeType::Mean(id) | NodeType::Variance(id) => {
                let (mut shape, dtype) = Self::get_shape_dtype(asg, *id)?;
                // Don't remove dimension, set it to 1 to preserve tensor rank
                // for correct broadcasting.
                if !shape.is_empty() {
                    *shape.last_mut().unwrap() = 1;
                }
                Ok((shape, dtype))
            }

            NodeType::Transpose(id, axis1, axis2) => {
                let (mut shape, dtype) = Self::get_shape_dtype(asg, *id)?;
                if *axis1 >= shape.len() || *axis2 >= shape.len() {
                    return Err(ShapeInferenceError::InvalidRank {
                        node_id: node.id,
                        expected: axis1.max(axis2) + 1,
                        actual: shape.len(),
                    });
                }
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

            NodeType::Softmax(id) => Self::get_shape_dtype(asg, *id),

            NodeType::GreaterThan(l, r) => {
                let (ls, _) = Self::get_shape_dtype(asg, *l)?;
                let (rs, _) = Self::get_shape_dtype(asg, *r)?;
                let out_shape = if ls.iter().product::<usize>() >= rs.iter().product::<usize>() { ls } else { rs };
                Ok((out_shape, DType::F32)) // Returns 0.0 or 1.0, so F32
            }

            NodeType::ReduceSumTo(source_id, target_shape_provider_id) => {
                let (_, dtype) = Self::get_shape_dtype(asg, *source_id)?;
                let (target_shape, _) = Self::get_shape_dtype(asg, *target_shape_provider_id)?;
                Ok((target_shape, dtype))
            }

            NodeType::MaxPool2d { input, kernel_size, stride } => {
                let (input_shape, dtype) = Self::get_shape_dtype(asg, *input)?;

                if input_shape.len() != 4 {
                    return Err(ShapeInferenceError::InvalidRank {
                        node_id: node.id,
                        expected: 4, // Expected [N, C, H, W]
                        actual: input_shape.len(),
                    });
                }

                let n = input_shape[0];
                let c = input_shape[1];
                let h = input_shape[2];
                let w = input_shape[3];

                if h < kernel_size.0 || w < kernel_size.1 {
                    return Err(ShapeInferenceError::MissingShapeInfo(node.id));
                }

                let out_h = (h - kernel_size.0) / stride.0 + 1;
                let out_w = (w - kernel_size.1) / stride.1 + 1;

                let output_shape = vec![n, c, out_h, out_w];
                Ok((output_shape, dtype))
            }

            NodeType::MaxUnpool2d { original_input, .. } => {
                // Unpooling output shape always matches the ORIGINAL pooling input shape.
                Self::get_shape_dtype(asg, *original_input)
            }

            NodeType::Power(base_id, _power_id) => {
                // Shape is determined by base
                Self::get_shape_dtype(asg, *base_id)
            }

            // Conv2d: [N, C_in, H, W] -> [N, C_out, H_out, W_out]
            NodeType::Conv2d { input, weight, stride, padding, dilation, groups, .. } => {
                let (input_shape, dtype) = Self::get_shape_dtype(asg, *input)?;
                let (weight_shape, _) = Self::get_shape_dtype(asg, *weight)?;

                if input_shape.len() != 4 {
                    return Err(ShapeInferenceError::InvalidRank {
                        node_id: node.id,
                        expected: 4,
                        actual: input_shape.len(),
                    });
                }
                if weight_shape.len() != 4 {
                    return Err(ShapeInferenceError::InvalidRank {
                        node_id: node.id,
                        expected: 4,
                        actual: weight_shape.len(),
                    });
                }

                let n = input_shape[0];
                let h = input_shape[2];
                let w = input_shape[3];

                let out_channels = weight_shape[0];
                let kernel_h = weight_shape[2];
                let kernel_w = weight_shape[3];

                // Effective kernel size with dilation
                let eff_kh = (kernel_h - 1) * dilation.0 + 1;
                let eff_kw = (kernel_w - 1) * dilation.1 + 1;

                let out_h = (h + 2 * padding.0 - eff_kh) / stride.0 + 1;
                let out_w = (w + 2 * padding.1 - eff_kw) / stride.1 + 1;

                Ok((vec![n, out_channels, out_h, out_w], dtype))
            }

            // ConvTranspose2d: [N, C_in, H, W] -> [N, C_out, H_out, W_out]
            NodeType::ConvTranspose2d { input, weight, stride, padding, output_padding, dilation, groups, .. } => {
                let (input_shape, dtype) = Self::get_shape_dtype(asg, *input)?;
                let (weight_shape, _) = Self::get_shape_dtype(asg, *weight)?;

                if input_shape.len() != 4 {
                    return Err(ShapeInferenceError::InvalidRank {
                        node_id: node.id,
                        expected: 4,
                        actual: input_shape.len(),
                    });
                }

                let n = input_shape[0];
                let h = input_shape[2];
                let w = input_shape[3];

                let out_channels_per_group = weight_shape[1];
                let out_channels = out_channels_per_group * groups;
                let kernel_h = weight_shape[2];
                let kernel_w = weight_shape[3];

                let out_h = (h - 1) * stride.0 - 2 * padding.0 + dilation.0 * (kernel_h - 1) + output_padding.0 + 1;
                let out_w = (w - 1) * stride.1 - 2 * padding.1 + dilation.1 * (kernel_w - 1) + output_padding.1 + 1;

                Ok((vec![n, out_channels, out_h, out_w], dtype))
            }

            // AvgPool2d: [N, C, H, W] -> [N, C, H_out, W_out]
            NodeType::AvgPool2d { input, kernel_size, stride, padding } => {
                let (input_shape, dtype) = Self::get_shape_dtype(asg, *input)?;

                if input_shape.len() != 4 {
                    return Err(ShapeInferenceError::InvalidRank {
                        node_id: node.id,
                        expected: 4,
                        actual: input_shape.len(),
                    });
                }

                let n = input_shape[0];
                let c = input_shape[1];
                let h = input_shape[2];
                let w = input_shape[3];

                let out_h = (h + 2 * padding.0 - kernel_size.0) / stride.0 + 1;
                let out_w = (w + 2 * padding.1 - kernel_size.1) / stride.1 + 1;

                Ok((vec![n, c, out_h, out_w], dtype))
            }

            // AdaptiveAvgPool2d: [N, C, H, W] -> [N, C, H_out, W_out]
            NodeType::AdaptiveAvgPool2d { input, output_size } => {
                let (input_shape, dtype) = Self::get_shape_dtype(asg, *input)?;

                if input_shape.len() != 4 {
                    return Err(ShapeInferenceError::InvalidRank {
                        node_id: node.id,
                        expected: 4,
                        actual: input_shape.len(),
                    });
                }

                let n = input_shape[0];
                let c = input_shape[1];

                Ok((vec![n, c, output_size.0, output_size.1], dtype))
            }

            // Embedding: indices[*] + weight[num_embeddings, embedding_dim] -> [*, embedding_dim]
            NodeType::Embedding { indices, weight } => {
                let (indices_shape, _) = Self::get_shape_dtype(asg, *indices)?;
                let (weight_shape, dtype) = Self::get_shape_dtype(asg, *weight)?;

                if weight_shape.len() != 2 {
                    return Err(ShapeInferenceError::InvalidRank {
                        node_id: node.id,
                        expected: 2,
                        actual: weight_shape.len(),
                    });
                }

                let embedding_dim = weight_shape[1];

                // Output shape: indices_shape + [embedding_dim]
                let mut output_shape = indices_shape;
                output_shape.push(embedding_dim);

                Ok((output_shape, dtype))
            }

            // EmbeddingGrad: grad_output[*, embedding_dim] + indices[*] -> [num_embeddings, embedding_dim]
            NodeType::EmbeddingGrad { grad_output, num_embeddings, .. } => {
                let (grad_shape, dtype) = Self::get_shape_dtype(asg, *grad_output)?;
                let embedding_dim = *grad_shape.last().unwrap();

                Ok((vec![*num_embeddings, embedding_dim], dtype))
            }

            // AvgUnpool2d: output has same shape as original input
            NodeType::AvgUnpool2d { original_input, .. } => {
                let (orig_shape, dtype) = Self::get_shape_dtype(asg, *original_input)?;
                Ok((orig_shape, dtype))
            }

            // Conv2dBackwardInput: output has same shape as original input
            NodeType::Conv2dBackwardInput { input_shape, grad_output, .. } => {
                let (_, dtype) = Self::get_shape_dtype(asg, *grad_output)?;
                let (n, c, h, w) = *input_shape;
                Ok((vec![n, c, h, w], dtype))
            }

            // Conv2dBackwardWeight: output has same shape as weight
            NodeType::Conv2dBackwardWeight { weight_shape, grad_output, .. } => {
                let (_, dtype) = Self::get_shape_dtype(asg, *grad_output)?;
                let (c_out, c_in, kh, kw) = *weight_shape;
                Ok((vec![c_out, c_in, kh, kw], dtype))
            }

            // LayerNorm: output has same shape as input
            NodeType::LayerNorm { input, .. } => {
                Self::get_shape_dtype(asg, *input)
            }

            // LayerNormBackward: output (gradient w.r.t. input) has same shape as input
            NodeType::LayerNormBackward { input, .. } => {
                Self::get_shape_dtype(asg, *input)
            }

            // LayerNormGradGamma: output has shape [1, norm_size] where norm_size is last dim of input
            NodeType::LayerNormGradGamma { input, .. } => {
                let (input_shape, dtype) = Self::get_shape_dtype(asg, *input)?;
                let norm_size = *input_shape.last().unwrap_or(&1);
                Ok((vec![1, norm_size], dtype))
            }

            // LayerNormGradBeta: output has shape [1, norm_size] where norm_size is last dim of grad_output
            NodeType::LayerNormGradBeta { grad_output } => {
                let (grad_shape, dtype) = Self::get_shape_dtype(asg, *grad_output)?;
                let norm_size = *grad_shape.last().unwrap_or(&1);
                Ok((vec![1, norm_size], dtype))
            }

            // --- Explicitly handle remaining nodes to avoid fallback ---
            unimplemented_type => Err(ShapeInferenceError::UnimplementedNodeType(format!("{:?}", unimplemented_type))),
        }
    }

    /// Helper function to get already computed shape and type for a node.
    fn get_shape_dtype(asg: &Asg, node_id: NodeId) -> Result<(Shape, DType)> {
        let node = asg.get_node(node_id)?;
        match (&node.shape, &node.dtype) {
            (Some(s), Some(d)) => Ok((s.clone(), *d)),
            _ => Err(ShapeInferenceError::MissingShapeInfo(node_id)),
        }
    }

    /// Performs topological sort of the graph.
    /// Returns a vector of node IDs in order suitable for computation.
    pub fn topological_sort(asg: &Asg) -> Result<Vec<NodeId>> {
        let mut sorted = Vec::new();
        let mut visited = HashSet::new();
        // IMPORTANT: need to traverse all outputs, not just one
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
        if visited.contains(&node_id) {
            return Ok(());
        }

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
            | NodeType::Reshape(a, b)
            | NodeType::ReduceSumTo(a, b) => vec![*a, *b],

            NodeType::ReLU(a)
            | NodeType::Sum(a)
            | NodeType::Sigmoid(a)
            | NodeType::Softmax(a)
            | NodeType::Mean(a)
            | NodeType::Variance(a)
            | NodeType::Sqrt(a)
            | NodeType::Log(a)
            | NodeType::Exp(a)
            | NodeType::Abs(a)
            | NodeType::Neg(a)
            | NodeType::Tanh(a)
            | NodeType::GELU(a)
            | NodeType::SiLU(a) => vec![*a],

            NodeType::LeakyReLU(a, _)
            | NodeType::ELU(a, _)
            | NodeType::Softplus(a, _)
            | NodeType::Clamp(a, _, _) => vec![*a],

            NodeType::Transpose(a, _, _) => vec![*a],
            NodeType::MaxPool2d { input, .. } => vec![*input],
            NodeType::MaxUnpool2d { input, original_input, .. } => vec![*input, *original_input],
            NodeType::Conv2d { input, weight, bias, .. } => {
                let mut deps = vec![*input, *weight];
                if let Some(b) = bias {
                    deps.push(*b);
                }
                deps
            }
            NodeType::ConvTranspose2d { input, weight, bias, .. } => {
                let mut deps = vec![*input, *weight];
                if let Some(b) = bias {
                    deps.push(*b);
                }
                deps
            }
            NodeType::AvgPool2d { input, .. } => vec![*input],
            NodeType::AdaptiveAvgPool2d { input, .. } => vec![*input],
            NodeType::Embedding { indices, weight } => vec![*indices, *weight],
            NodeType::EmbeddingGrad { grad_output, indices, .. } => vec![*grad_output, *indices],
            NodeType::AvgUnpool2d { input, original_input, .. } => vec![*input, *original_input],
            NodeType::Conv2dBackwardInput { grad_output, weight, .. } => vec![*grad_output, *weight],
            NodeType::Conv2dBackwardWeight { grad_output, input, .. } => vec![*grad_output, *input],
            NodeType::LayerNorm { input, gamma, beta, .. } => vec![*input, *gamma, *beta],
            NodeType::LayerNormBackward { grad_output, input, gamma, .. } => vec![*grad_output, *input, *gamma],
            NodeType::LayerNormGradGamma { grad_output, input, .. } => vec![*grad_output, *input],
            NodeType::LayerNormGradBeta { grad_output } => vec![*grad_output],
            _ => vec![],
        };

        for input_id in inputs {
            Self::build_sorted_graph(input_id, asg, visited, sorted)?;
        }

        if !visited.contains(&node_id) {
            visited.insert(node_id);
            sorted.push(node_id);
        }
        Ok(())
    }
}
