//! XOR Example - Training a simple neural network to learn the XOR function.
//!
//! This example demonstrates:
//! - Creating a graph context and building a neural network
//! - Defining inputs, parameters, and forward pass
//! - Computing loss and gradients using autograd
//! - Training with SGD optimizer
//!
//! Run with: `cargo run --example xor`

use ndarray::{array, ArrayD, IxDyn};
use rustyasg::analysis::shape_inference::ShapeInference;
use rustyasg::asg::{DType, Value};
use rustyasg::autograd::Gradients;
use rustyasg::losses::mse_loss;
use rustyasg::nn::{Linear, Module};
use rustyasg::optimizers::{Optimizer, Sgd};
use rustyasg::runtime::{backend::Backend, cpu_backend::CpuBackend};
use rustyasg::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

fn main() {
    println!("=== RustyASG XOR Example ===\n");

    // XOR truth table
    // Input: [0,0] -> Output: 0
    // Input: [0,1] -> Output: 1
    // Input: [1,0] -> Output: 1
    // Input: [1,1] -> Output: 0

    let x_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]].into_dyn();
    let y_data = array![[0.0], [1.0], [1.0], [0.0]].into_dyn();

    // Create graph context
    let context = Rc::new(RefCell::new(GraphContext::new()));

    // Define inputs
    let x = Tensor::new_input(&context, "x");
    let y_true = Tensor::new_input(&context, "y_true");

    // Build a simple 2-layer neural network: 2 -> 8 -> 1
    let layer1 = Linear::new(&context, "layer1"); // 2 -> 8
    let layer2 = Linear::new(&context, "layer2"); // 8 -> 1

    // Forward pass
    let hidden = layer1.forward(&x).relu();
    let y_pred = layer2.forward(&hidden).sigmoid();

    // Compute loss
    let loss = mse_loss(&y_pred, &y_true);

    // Set loss as output for forward graph
    context.borrow_mut().main_graph_mut().set_output(loss.node_id);

    // Get parameter node IDs for gradient computation
    let params: Vec<Tensor> = [layer1.parameters(), layer2.parameters()].concat();
    let param_ids: Vec<_> = params.iter().map(|p| p.node_id).collect();

    // Run shape inference before building gradients
    let mut initial_shapes = HashMap::new();
    initial_shapes.insert("x".to_string(), (vec![4, 2], DType::F32)); // 4 samples, 2 features
    initial_shapes.insert("y_true".to_string(), (vec![4, 1], DType::F32)); // 4 samples, 1 output
    initial_shapes.insert("layer1.weights".to_string(), (vec![2, 8], DType::F32));
    initial_shapes.insert("layer1.bias".to_string(), (vec![1, 8], DType::F32));
    initial_shapes.insert("layer2.weights".to_string(), (vec![8, 1], DType::F32));
    initial_shapes.insert("layer2.bias".to_string(), (vec![1, 1], DType::F32));

    let mut forward_graph = context.borrow().main_graph().clone();
    ShapeInference::run(&mut forward_graph, &initial_shapes).expect("Shape inference failed");

    // Compute gradients using autograd
    println!("Building gradient graph...");
    let grad_graph = Gradients::new(forward_graph.clone())
        .build(loss.node_id, &param_ids)
        .expect("Failed to build gradient graph");

    println!(
        "Forward graph: {} nodes, Gradient graph: {} nodes\n",
        context.borrow().main_graph().nodes.len(),
        grad_graph.nodes.len()
    );

    // Initialize parameters with random values
    // Layer1: weights [2, 8], bias [8]
    // Layer2: weights [8, 1], bias [1]
    let mut param_values: HashMap<String, Value> = HashMap::new();

    // Initialize with small random values (using simple deterministic "random")
    let w1: Vec<f32> = (0..16).map(|i| ((i as f32 * 0.1).sin() * 0.5)).collect();
    let b1: Vec<f32> = vec![0.0; 8];
    let w2: Vec<f32> = (0..8).map(|i| ((i as f32 * 0.2).cos() * 0.5)).collect();
    let b2: Vec<f32> = vec![0.0; 1];

    param_values.insert(
        "layer1.weights".to_string(),
        Value::Tensor(ArrayD::from_shape_vec(IxDyn(&[2, 8]), w1).unwrap()),
    );
    param_values.insert(
        "layer1.bias".to_string(),
        Value::Tensor(ArrayD::from_shape_vec(IxDyn(&[1, 8]), b1).unwrap()),
    );
    param_values.insert(
        "layer2.weights".to_string(),
        Value::Tensor(ArrayD::from_shape_vec(IxDyn(&[8, 1]), w2).unwrap()),
    );
    param_values.insert(
        "layer2.bias".to_string(),
        Value::Tensor(ArrayD::from_shape_vec(IxDyn(&[1, 1]), b2).unwrap()),
    );

    // Create optimizer
    let mut optimizer = Sgd::new(0.5).with_momentum(0.9);

    // Create backend
    let backend = CpuBackend::new();

    // Training loop
    let epochs = 1000;
    println!("Starting training for {} epochs...\n", epochs);

    for epoch in 0..epochs {
        // Prepare input data
        let mut inputs = param_values.clone();
        inputs.insert("x".to_string(), Value::Tensor(x_data.clone()));
        inputs.insert("y_true".to_string(), Value::Tensor(y_data.clone()));

        // Load data to device
        let device_data = backend.load_data(&inputs).unwrap();

        // Build memo for forward pass
        let mut memo = HashMap::new();
        for (name, value) in device_data {
            let node_id = forward_graph
                .nodes
                .iter()
                .find(|(_, node)| {
                    matches!(
                        &node.node_type,
                        rustyasg::asg::NodeType::Input { name: n } |
                        rustyasg::asg::NodeType::Parameter { name: n } if n == &name
                    )
                })
                .map(|(id, _)| *id);

            if let Some(id) = node_id {
                memo.insert((forward_graph.id, id), value);
            }
        }

        // Run forward pass
        let (forward_results, forward_memo) = backend.run(&forward_graph, memo).unwrap();

        // Get loss value
        let loss_value = match &forward_results[0] {
            Value::Tensor(arr) => arr.iter().sum::<f32>(),
            _ => panic!("Expected tensor"),
        };

        // Run gradient graph
        let (grad_results, _) = backend.run(&grad_graph, forward_memo).unwrap();

        // Extract gradients
        let mut gradients: HashMap<String, Value> = HashMap::new();
        for (i, grad_output_id) in grad_graph.outputs.iter().enumerate() {
            // Find corresponding parameter name
            let param_name = match i {
                0 => "layer1.weights",
                1 => "layer1.bias",
                2 => "layer2.weights",
                3 => "layer2.bias",
                _ => continue,
            };
            gradients.insert(param_name.to_string(), grad_results[i].clone());
        }

        // Update parameters
        optimizer.step(&mut param_values, &gradients);

        // Print progress
        if epoch % 100 == 0 || epoch == epochs - 1 {
            println!("Epoch {:4}: Loss = {:.6}", epoch, loss_value);
        }
    }

    // Final evaluation
    println!("\n=== Final Evaluation ===\n");

    let mut inputs = param_values.clone();
    inputs.insert("x".to_string(), Value::Tensor(x_data.clone()));
    inputs.insert("y_true".to_string(), Value::Tensor(y_data.clone()));

    let device_data = backend.load_data(&inputs).unwrap();
    let mut memo = HashMap::new();
    for (name, value) in device_data {
        let node_id = context
            .borrow()
            .main_graph()
            .nodes
            .iter()
            .find(|(_, node)| {
                matches!(
                    &node.node_type,
                    rustyasg::asg::NodeType::Input { name: n } |
                    rustyasg::asg::NodeType::Parameter { name: n } if n == &name
                )
            })
            .map(|(id, _)| *id);

        if let Some(id) = node_id {
            memo.insert((context.borrow().main_graph().id, id), value);
        }
    }

    // We need to get predictions, not just loss
    // For this, let's create a new context for inference
    let inference_context = Rc::new(RefCell::new(GraphContext::new()));
    let x_inf = Tensor::new_input(&inference_context, "x");
    let layer1_inf = Linear::new(&inference_context, "layer1");
    let layer2_inf = Linear::new(&inference_context, "layer2");
    let hidden_inf = layer1_inf.forward(&x_inf).relu();
    let pred_inf = layer2_inf.forward(&hidden_inf).sigmoid();
    inference_context
        .borrow_mut()
        .main_graph_mut()
        .set_output(pred_inf.node_id);

    let mut inf_inputs = param_values.clone();
    inf_inputs.insert("x".to_string(), Value::Tensor(x_data.clone()));

    let inf_device_data = backend.load_data(&inf_inputs).unwrap();
    let mut inf_memo = HashMap::new();
    for (name, value) in inf_device_data {
        let node_id = inference_context
            .borrow()
            .main_graph()
            .nodes
            .iter()
            .find(|(_, node)| {
                matches!(
                    &node.node_type,
                    rustyasg::asg::NodeType::Input { name: n } |
                    rustyasg::asg::NodeType::Parameter { name: n } if n == &name
                )
            })
            .map(|(id, _)| *id);

        if let Some(id) = node_id {
            inf_memo.insert((inference_context.borrow().main_graph().id, id), value);
        }
    }

    let inf_graph = inference_context.borrow().main_graph().clone();
    let (inf_results, _) = backend.run(&inf_graph, inf_memo).unwrap();

    if let Value::Tensor(predictions) = &inf_results[0] {
        let inputs_arr = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let targets = [0.0, 1.0, 1.0, 0.0];

        println!("Input      | Target | Prediction | Rounded");
        println!("-----------|--------|------------|--------");
        for i in 0..4 {
            let pred = predictions[[i, 0]];
            let rounded = if pred > 0.5 { 1.0 } else { 0.0 };
            let correct = if rounded == targets[i] { "✓" } else { "✗" };
            println!(
                "[{:.0}, {:.0}]     |   {:.0}    |   {:.4}    |   {:.0}  {}",
                inputs_arr[i][0], inputs_arr[i][1], targets[i], pred, rounded, correct
            );
        }
    }

    println!("\n=== Training Complete ===");
}
