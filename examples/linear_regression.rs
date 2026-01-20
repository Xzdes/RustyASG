//! Linear Regression Example - The simplest possible training example.
//!
//! This example demonstrates the basic training loop:
//! 1. Create computation graph
//! 2. Run forward pass
//! 3. Compute gradients
//! 4. Update parameters
//!
//! We learn the function y = 2*x + 1
//!
//! Run with: `cargo run --example linear_regression`

use ndarray::{ArrayD, IxDyn};
use rustyasg::analysis::shape_inference::ShapeInference;
use rustyasg::asg::{DType, Value};
use rustyasg::autograd::Gradients;
use rustyasg::losses::mse_loss_mean;
use rustyasg::optimizers::{Optimizer, Sgd};
use rustyasg::runtime::{backend::Backend, cpu_backend::CpuBackend};
use rustyasg::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

fn main() {
    println!("=== RustyASG Linear Regression Example ===\n");
    println!("Learning the function: y = 2*x + 1\n");

    // Generate training data: y = 2*x + 1
    let x_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
    let y_data: Vec<f32> = x_data.iter().map(|&x| 2.0 * x + 1.0).collect();

    let x_tensor = ArrayD::from_shape_vec(IxDyn(&[10, 1]), x_data.clone()).unwrap();
    let y_tensor = ArrayD::from_shape_vec(IxDyn(&[10, 1]), y_data.clone()).unwrap();

    // Create graph context
    let context = Rc::new(RefCell::new(GraphContext::new()));

    // Define inputs and parameters
    let x = Tensor::new_input(&context, "x");
    let y_true = Tensor::new_input(&context, "y_true");
    let w = Tensor::new_parameter(&context, "w"); // weight (slope)
    let b = Tensor::new_parameter(&context, "b"); // bias (intercept)

    // Forward pass: y_pred = x * w + b
    let y_pred = &(&x * &w) + &b;

    // Compute loss (use mean for stable gradients)
    let loss = mse_loss_mean(&y_pred, &y_true);

    // Set loss as output
    context.borrow_mut().main_graph_mut().set_output(loss.node_id);

    // Run shape inference before building gradients
    let mut initial_shapes = HashMap::new();
    initial_shapes.insert("x".to_string(), (vec![10, 1], DType::F32));
    initial_shapes.insert("y_true".to_string(), (vec![10, 1], DType::F32));
    initial_shapes.insert("w".to_string(), (vec![1, 1], DType::F32));
    initial_shapes.insert("b".to_string(), (vec![1, 1], DType::F32));

    let mut forward_graph = context.borrow().main_graph().clone();
    ShapeInference::run(&mut forward_graph, &initial_shapes).expect("Shape inference failed");

    // Build gradient graph
    println!("Building gradient graph...");
    let grad_graph = Gradients::new(forward_graph.clone())
        .build(loss.node_id, &[w.node_id, b.node_id])
        .expect("Failed to build gradient graph");

    println!(
        "Forward graph: {} nodes, Gradient graph: {} nodes\n",
        context.borrow().main_graph().nodes.len(),
        grad_graph.nodes.len()
    );

    // Initialize parameters
    let mut param_values: HashMap<String, Value> = HashMap::new();
    param_values.insert(
        "w".to_string(),
        Value::Tensor(ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.0]).unwrap()),
    );
    param_values.insert(
        "b".to_string(),
        Value::Tensor(ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![0.0]).unwrap()),
    );

    // Create optimizer and backend
    // Use small learning rate due to large x values (0-9)
    let mut optimizer = Sgd::new(0.002).with_momentum(0.9);
    let backend = CpuBackend::new();

    // Training loop
    let epochs = 200;
    println!("Training for {} epochs...\n", epochs);

    for epoch in 0..epochs {
        // Prepare inputs
        let mut inputs = param_values.clone();
        inputs.insert("x".to_string(), Value::Tensor(x_tensor.clone()));
        inputs.insert("y_true".to_string(), Value::Tensor(y_tensor.clone()));

        // Load data
        let device_data = backend.load_data(&inputs).unwrap();

        // Build memo
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
        gradients.insert("w".to_string(), grad_results[0].clone());
        gradients.insert("b".to_string(), grad_results[1].clone());

        // Update parameters
        optimizer.step(&mut param_values, &gradients);

        // Print progress
        if epoch % 10 == 0 || epoch == epochs - 1 {
            let w_val = match &param_values["w"] {
                Value::Tensor(arr) => arr[[0, 0]],
                _ => 0.0,
            };
            let b_val = match &param_values["b"] {
                Value::Tensor(arr) => arr[[0, 0]],
                _ => 0.0,
            };
            println!(
                "Epoch {:3}: Loss = {:8.4}, w = {:6.3}, b = {:6.3}",
                epoch, loss_value, w_val, b_val
            );
        }
    }

    // Final results
    println!("\n=== Results ===\n");
    let w_final = match &param_values["w"] {
        Value::Tensor(arr) => arr[[0, 0]],
        _ => 0.0,
    };
    let b_final = match &param_values["b"] {
        Value::Tensor(arr) => arr[[0, 0]],
        _ => 0.0,
    };

    println!("Target function:  y = 2.000*x + 1.000");
    println!("Learned function: y = {:.3}*x + {:.3}", w_final, b_final);
    println!("\nError in w: {:.4}", (w_final - 2.0).abs());
    println!("Error in b: {:.4}", (b_final - 1.0).abs());

    println!("\n=== Training Complete ===");
}
