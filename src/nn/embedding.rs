//! Module with Embedding layer implementation.
//!
//! Embedding layer transforms integer indices into dense vectors
//! of fixed dimensionality. Widely used in NLP for representing
//! words, tokens, and other discrete entities.

use super::module::Module;
use crate::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Embedding layer for transforming indices into dense vectors.
///
/// # Usage Example
///
/// ```ignore
/// use rustyasg::nn::Embedding;
///
/// let context = Rc::new(RefCell::new(GraphContext::new()));
/// let embedding = Embedding::new(&context, 10000, 256, "token_embedding");
/// // vocab_size=10000, embedding_dim=256
///
/// let indices = Tensor::new_input(&context, "token_ids"); // [batch, seq_len]
/// let embedded = embedding.forward(&indices); // [batch, seq_len, 256]
/// ```
#[derive(Debug, Clone)]
pub struct Embedding {
    /// Number of unique indices (vocabulary size).
    pub num_embeddings: usize,
    /// Embedding vector dimensionality.
    pub embedding_dim: usize,
    /// Embedding matrix of shape [num_embeddings, embedding_dim].
    pub weight: Tensor,
}

impl Embedding {
    /// Creates a new Embedding layer.
    ///
    /// # Arguments
    ///
    /// * `context` - Graph context
    /// * `num_embeddings` - Vocabulary size (number of unique indices)
    /// * `embedding_dim` - Output vector dimensionality
    /// * `name` - Name for parameters (used as prefix)
    ///
    /// # Returns
    ///
    /// New `Embedding` instance with weights of shape [num_embeddings, embedding_dim]
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        num_embeddings: usize,
        embedding_dim: usize,
        name: &str,
    ) -> Self {
        let weight = Tensor::new_parameter(context, &format!("{}_weight", name));

        Self {
            num_embeddings,
            embedding_dim,
            weight,
        }
    }

    /// Creates Embedding layer with provided weight tensor.
    ///
    /// Useful for loading pretrained embeddings.
    pub fn from_weight(weight: Tensor, num_embeddings: usize, embedding_dim: usize) -> Self {
        Self {
            num_embeddings,
            embedding_dim,
            weight,
        }
    }
}

impl Module for Embedding {
    /// Performs embedding lookup.
    ///
    /// # Arguments
    ///
    /// * `indices` - Index tensor of any shape [*]
    ///
    /// # Returns
    ///
    /// Tensor of shape [*, embedding_dim]
    fn forward(&self, input: &Tensor) -> Tensor {
        input.embedding(&self.weight)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::asg::Value;
    use crate::runtime::{backend::Backend, cpu_backend::CpuBackend};
    use ndarray::{arr2, ArrayD, IxDyn};
    use std::collections::HashMap;

    #[test]
    fn test_embedding_creation() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let embedding = Embedding::new(&context, 1000, 64, "emb");

        assert_eq!(embedding.num_embeddings, 1000);
        assert_eq!(embedding.embedding_dim, 64);
        assert_eq!(embedding.parameters().len(), 1);
    }

    #[test]
    fn test_embedding_forward() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let embedding = Embedding::new(&context, 5, 3, "emb");

        // Create input indices
        let indices = Tensor::new_input(&context, "indices");

        // Forward pass
        let output = embedding.forward(&indices);

        // Set graph output
        context.borrow_mut().main_graph_mut().set_output(output.node_id);

        // Prepare data
        // Embedding weights: 5x3 matrix
        let weight_data = arr2(&[
            [1.0, 2.0, 3.0],    // index 0
            [4.0, 5.0, 6.0],    // index 1
            [7.0, 8.0, 9.0],    // index 2
            [10.0, 11.0, 12.0], // index 3
            [13.0, 14.0, 15.0], // index 4
        ]).into_dyn();

        // Indices: [2, 3] -> select index 0 and index 2
        let indices_data = ArrayD::from_shape_vec(IxDyn(&[2]), vec![0.0, 2.0]).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("indices".to_string(), Value::Tensor(indices_data));
        inputs.insert("emb_weight".to_string(), Value::Tensor(weight_data));

        // Run
        let backend = CpuBackend::new();
        let device_data = backend.load_data(&inputs).unwrap();

        let mut memo = HashMap::new();
        for (name, value) in device_data {
            let node_id = context
                .borrow()
                .main_graph()
                .nodes
                .iter()
                .find(|(_, node)| {
                    matches!(&node.node_type,
                        crate::asg::NodeType::Input { name: n } |
                        crate::asg::NodeType::Parameter { name: n } if n == &name
                    )
                })
                .map(|(id, _)| *id);

            if let Some(id) = node_id {
                memo.insert((0, id), value);
            }
        }

        let graph = context.borrow().main_graph().clone();
        let (results, _) = backend.run(&graph, memo).unwrap();

        // Check result
        let result = &results[0];
        if let Value::Tensor(arr) = result {
            assert_eq!(arr.shape(), &[2, 3]); // [num_indices, embedding_dim]

            // index 0 -> [1, 2, 3]
            assert!((arr[[0, 0]] - 1.0).abs() < 1e-5);
            assert!((arr[[0, 1]] - 2.0).abs() < 1e-5);
            assert!((arr[[0, 2]] - 3.0).abs() < 1e-5);

            // index 2 -> [7, 8, 9]
            assert!((arr[[1, 0]] - 7.0).abs() < 1e-5);
            assert!((arr[[1, 1]] - 8.0).abs() < 1e-5);
            assert!((arr[[1, 2]] - 9.0).abs() < 1e-5);
        } else {
            panic!("Expected tensor output");
        }
    }

    #[test]
    fn test_embedding_2d_indices() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let embedding = Embedding::new(&context, 5, 4, "emb");

        let indices = Tensor::new_input(&context, "indices");
        let output = embedding.forward(&indices);

        context.borrow_mut().main_graph_mut().set_output(output.node_id);

        // Embedding weights: 5x4
        let weight_data = ArrayD::from_shape_vec(
            IxDyn(&[5, 4]),
            (0..20).map(|x| x as f32).collect()
        ).unwrap();

        // Indices: [2, 3] (batch=2, seq_len=3)
        let indices_data = ArrayD::from_shape_vec(
            IxDyn(&[2, 3]),
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 0.0]
        ).unwrap();

        let mut inputs = HashMap::new();
        inputs.insert("indices".to_string(), Value::Tensor(indices_data));
        inputs.insert("emb_weight".to_string(), Value::Tensor(weight_data));

        let backend = CpuBackend::new();
        let device_data = backend.load_data(&inputs).unwrap();

        let mut memo = HashMap::new();
        for (name, value) in device_data {
            let node_id = context
                .borrow()
                .main_graph()
                .nodes
                .iter()
                .find(|(_, node)| {
                    matches!(&node.node_type,
                        crate::asg::NodeType::Input { name: n } |
                        crate::asg::NodeType::Parameter { name: n } if n == &name
                    )
                })
                .map(|(id, _)| *id);

            if let Some(id) = node_id {
                memo.insert((0, id), value);
            }
        }

        let graph = context.borrow().main_graph().clone();
        let (results, _) = backend.run(&graph, memo).unwrap();

        if let Value::Tensor(arr) = &results[0] {
            // Output shape: [2, 3, 4]
            assert_eq!(arr.shape(), &[2, 3, 4]);
        } else {
            panic!("Expected tensor output");
        }
    }
}
