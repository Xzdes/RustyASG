//! MNIST Digit Classification Example
//!
//! This example demonstrates training a neural network on MNIST dataset.
//! Since MNIST data is large, this example uses a synthetic "mini-MNIST" dataset
//! for demonstration purposes.
//!
//! For real MNIST training, download the dataset and load it using the data pipeline.
//!
//! Network architecture: 784 -> 128 -> 64 -> 10
//!
//! Run with: `cargo run --example mnist`

use ndarray::{ArrayD, IxDyn};
use rustyasg::analysis::shape_inference::ShapeInference;
use rustyasg::asg::{DType, Value};
use rustyasg::autograd::Gradients;
use rustyasg::losses::mse_loss_mean;
use rustyasg::nn::{Linear, Module};
use rustyasg::optimizers::{Adam, Optimizer};
use rustyasg::runtime::{backend::Backend, cpu_backend::CpuBackend};
use rustyasg::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Generate synthetic MNIST-like data for demonstration.
/// In production, load real MNIST from files.
fn generate_synthetic_mnist(num_samples: usize) -> (ArrayD<f32>, ArrayD<f32>) {
    let mut images = vec![0.0f32; num_samples * 784];
    let mut labels = vec![0.0f32; num_samples * 10];

    for i in 0..num_samples {
        let digit = i % 10;

        // Create a simple pattern for each digit
        // (In real MNIST these would be actual handwritten digits)
        for j in 0..784 {
            // Create distinct patterns for each digit
            let row = j / 28;
            let col = j % 28;

            let value = match digit {
                0 => {
                    // Circle pattern
                    let center = 14.0;
                    let dist = ((row as f32 - center).powi(2) + (col as f32 - center).powi(2)).sqrt();
                    if dist > 8.0 && dist < 12.0 { 1.0 } else { 0.0 }
                }
                1 => {
                    // Vertical line
                    if col >= 12 && col <= 16 { 1.0 } else { 0.0 }
                }
                2 => {
                    // Top half horizontal, then diagonal
                    if row < 10 && col > 8 && col < 20 { 1.0 }
                    else if row >= 10 && row < 18 && col == 20 - (row - 10) { 1.0 }
                    else if row >= 18 && col > 8 && col < 20 { 1.0 }
                    else { 0.0 }
                }
                3 => {
                    // Three horizontal lines
                    if (row == 6 || row == 14 || row == 22) && col > 8 && col < 20 { 1.0 }
                    else if col == 19 && row > 6 && row < 22 { 1.0 }
                    else { 0.0 }
                }
                4 => {
                    // L shape rotated
                    if col == 8 && row < 16 { 1.0 }
                    else if row == 14 && col > 8 && col < 20 { 1.0 }
                    else if col == 18 { 1.0 }
                    else { 0.0 }
                }
                5 => {
                    // S pattern
                    if row == 6 && col > 8 && col < 20 { 1.0 }
                    else if row > 6 && row < 14 && col == 8 { 1.0 }
                    else if row == 14 && col > 8 && col < 20 { 1.0 }
                    else if row > 14 && row < 22 && col == 19 { 1.0 }
                    else if row == 22 && col > 8 && col < 20 { 1.0 }
                    else { 0.0 }
                }
                6 => {
                    // 6 pattern
                    let center = 14.0;
                    let dist = ((row as f32 - 18.0).powi(2) + (col as f32 - center).powi(2)).sqrt();
                    if dist > 4.0 && dist < 7.0 { 1.0 }
                    else if col == 8 && row > 6 && row < 18 { 1.0 }
                    else if row == 6 && col > 8 && col < 20 { 1.0 }
                    else { 0.0 }
                }
                7 => {
                    // 7 pattern
                    if row == 6 && col > 8 && col < 20 { 1.0 }
                    else if row > 6 && col == 18usize.saturating_sub((row - 6) / 2) { 1.0 }
                    else { 0.0 }
                }
                8 => {
                    // 8 pattern - two circles
                    let dist1 = ((row as f32 - 10.0).powi(2) + (col as f32 - 14.0).powi(2)).sqrt();
                    let dist2 = ((row as f32 - 18.0).powi(2) + (col as f32 - 14.0).powi(2)).sqrt();
                    if (dist1 > 3.0 && dist1 < 5.0) || (dist2 > 3.0 && dist2 < 5.0) { 1.0 } else { 0.0 }
                }
                9 => {
                    // 9 pattern
                    let dist = ((row as f32 - 10.0).powi(2) + (col as f32 - 14.0).powi(2)).sqrt();
                    if dist > 4.0 && dist < 7.0 { 1.0 }
                    else if col == 19 && row > 10 && row < 24 { 1.0 }
                    else { 0.0 }
                }
                _ => 0.0,
            };

            images[i * 784 + j] = value;
        }

        // One-hot encode label
        labels[i * 10 + digit] = 1.0;
    }

    let images_arr = ArrayD::from_shape_vec(IxDyn(&[num_samples, 784]), images).unwrap();
    let labels_arr = ArrayD::from_shape_vec(IxDyn(&[num_samples, 10]), labels).unwrap();

    (images_arr, labels_arr)
}

fn main() {
    println!("=== RustyASG MNIST Example ===\n");

    // Generate synthetic data (replace with real MNIST in production)
    let num_train = 100;
    let num_test = 20;
    println!("Generating synthetic MNIST data...");
    let (train_images, train_labels) = generate_synthetic_mnist(num_train);
    let (test_images, test_labels) = generate_synthetic_mnist(num_test);
    println!("Train samples: {}, Test samples: {}\n", num_train, num_test);

    // Create graph context
    let context = Rc::new(RefCell::new(GraphContext::new()));

    // Define inputs
    let x = Tensor::new_input(&context, "x");
    let y_true = Tensor::new_input(&context, "y_true");

    // Build network: 784 -> 128 -> 64 -> 10
    let layer1 = Linear::new(&context, "layer1"); // 784 -> 128
    let layer2 = Linear::new(&context, "layer2"); // 128 -> 64
    let layer3 = Linear::new(&context, "layer3"); // 64 -> 10

    // Forward pass with ReLU activations
    // Using sigmoid instead of softmax for simpler gradients
    let h1 = layer1.forward(&x).relu();
    let h2 = layer2.forward(&h1).relu();
    let logits = layer3.forward(&h2);
    let y_pred = logits.sigmoid();

    // MSE loss (using mean for stable gradients)
    let loss = mse_loss_mean(&y_pred, &y_true);

    // Set output
    context.borrow_mut().main_graph_mut().set_output(loss.node_id);

    // Get all parameters
    let params: Vec<Tensor> = [
        layer1.parameters(),
        layer2.parameters(),
        layer3.parameters(),
    ]
    .concat();
    let param_ids: Vec<_> = params.iter().map(|p| p.node_id).collect();

    // Run shape inference before building gradients
    let batch_size = 20;
    let mut initial_shapes = HashMap::new();
    initial_shapes.insert("x".to_string(), (vec![batch_size, 784], DType::F32));
    initial_shapes.insert("y_true".to_string(), (vec![batch_size, 10], DType::F32));
    initial_shapes.insert("layer1.weights".to_string(), (vec![784, 128], DType::F32));
    initial_shapes.insert("layer1.bias".to_string(), (vec![1, 128], DType::F32));
    initial_shapes.insert("layer2.weights".to_string(), (vec![128, 64], DType::F32));
    initial_shapes.insert("layer2.bias".to_string(), (vec![1, 64], DType::F32));
    initial_shapes.insert("layer3.weights".to_string(), (vec![64, 10], DType::F32));
    initial_shapes.insert("layer3.bias".to_string(), (vec![1, 10], DType::F32));

    let mut forward_graph = context.borrow().main_graph().clone();
    ShapeInference::run(&mut forward_graph, &initial_shapes).expect("Shape inference failed");

    // Build gradient graph
    println!("Building gradient graph...");
    let grad_graph = Gradients::new(forward_graph.clone())
        .build(loss.node_id, &param_ids)
        .expect("Failed to build gradient graph");

    println!(
        "Forward graph: {} nodes, Gradient graph: {} nodes\n",
        context.borrow().main_graph().nodes.len(),
        grad_graph.nodes.len()
    );

    // Initialize parameters
    let mut param_values: HashMap<String, Value> = HashMap::new();

    // Xavier initialization
    fn xavier_init(fan_in: usize, fan_out: usize) -> Vec<f32> {
        let scale = (6.0 / (fan_in + fan_out) as f32).sqrt();
        (0..fan_in * fan_out)
            .map(|i| {
                let x = (i as f32 * 0.618033988749895) % 1.0; // Golden ratio for pseudo-random
                (x * 2.0 - 1.0) * scale
            })
            .collect()
    }

    // Layer 1: 784 -> 128
    param_values.insert(
        "layer1.weights".to_string(),
        Value::Tensor(ArrayD::from_shape_vec(IxDyn(&[784, 128]), xavier_init(784, 128)).unwrap()),
    );
    param_values.insert(
        "layer1.bias".to_string(),
        Value::Tensor(ArrayD::zeros(IxDyn(&[1, 128]))),
    );

    // Layer 2: 128 -> 64
    param_values.insert(
        "layer2.weights".to_string(),
        Value::Tensor(ArrayD::from_shape_vec(IxDyn(&[128, 64]), xavier_init(128, 64)).unwrap()),
    );
    param_values.insert(
        "layer2.bias".to_string(),
        Value::Tensor(ArrayD::zeros(IxDyn(&[1, 64]))),
    );

    // Layer 3: 64 -> 10
    param_values.insert(
        "layer3.weights".to_string(),
        Value::Tensor(ArrayD::from_shape_vec(IxDyn(&[64, 10]), xavier_init(64, 10)).unwrap()),
    );
    param_values.insert(
        "layer3.bias".to_string(),
        Value::Tensor(ArrayD::zeros(IxDyn(&[1, 10]))),
    );

    // Create optimizer (lower learning rate for stability)
    let mut optimizer = Adam::new(0.0005);
    let backend = CpuBackend::new();

    // Training loop
    let epochs = 100;
    let num_batches = num_train / batch_size;

    println!("Training for {} epochs, batch size {}...\n", epochs, batch_size);

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = start + batch_size;

            // Get batch data
            let batch_images = train_images
                .slice(ndarray::s![start..end, ..])
                .to_owned()
                .into_dyn();
            let batch_labels = train_labels
                .slice(ndarray::s![start..end, ..])
                .to_owned()
                .into_dyn();

            // Prepare inputs
            let mut inputs = param_values.clone();
            inputs.insert("x".to_string(), Value::Tensor(batch_images));
            inputs.insert("y_true".to_string(), Value::Tensor(batch_labels));

            // Load and build memo
            let device_data = backend.load_data(&inputs).unwrap();
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

            // Forward pass
            let (forward_results, forward_memo) = backend.run(&forward_graph, memo).unwrap();

            let batch_loss = match &forward_results[0] {
                Value::Tensor(arr) => arr.iter().sum::<f32>(),
                _ => panic!("Expected tensor"),
            };
            epoch_loss += batch_loss;

            // Backward pass
            let (grad_results, _) = backend.run(&grad_graph, forward_memo).unwrap();

            // Extract gradients
            let param_names = [
                "layer1.weights", "layer1.bias",
                "layer2.weights", "layer2.bias",
                "layer3.weights", "layer3.bias",
            ];
            let mut gradients: HashMap<String, Value> = HashMap::new();
            for (i, name) in param_names.iter().enumerate() {
                if i < grad_results.len() {
                    gradients.insert(name.to_string(), grad_results[i].clone());
                }
            }

            // Update parameters
            optimizer.step(&mut param_values, &gradients);
        }

        // Print progress
        if epoch % 5 == 0 || epoch == epochs - 1 {
            let avg_loss = epoch_loss / num_batches as f32;
            println!("Epoch {:3}: Average Loss = {:.4}", epoch, avg_loss);
        }
    }

    // Evaluation on test set
    println!("\n=== Evaluation on Test Set ===\n");

    // Create inference context
    let inf_context = Rc::new(RefCell::new(GraphContext::new()));
    let x_inf = Tensor::new_input(&inf_context, "x");
    let layer1_inf = Linear::new(&inf_context, "layer1");
    let layer2_inf = Linear::new(&inf_context, "layer2");
    let layer3_inf = Linear::new(&inf_context, "layer3");

    let h1_inf = layer1_inf.forward(&x_inf).relu();
    let h2_inf = layer2_inf.forward(&h1_inf).relu();
    let logits_inf = layer3_inf.forward(&h2_inf);
    let pred_inf = logits_inf.sigmoid();

    inf_context.borrow_mut().main_graph_mut().set_output(pred_inf.node_id);

    // Run inference
    let mut inf_inputs = param_values.clone();
    inf_inputs.insert("x".to_string(), Value::Tensor(test_images.clone()));

    let inf_device_data = backend.load_data(&inf_inputs).unwrap();
    let mut inf_memo = HashMap::new();
    for (name, value) in inf_device_data {
        let node_id = inf_context
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
            inf_memo.insert((inf_context.borrow().main_graph().id, id), value);
        }
    }

    let inf_graph = inf_context.borrow().main_graph().clone();
    let (inf_results, _) = backend.run(&inf_graph, inf_memo).unwrap();

    // Calculate accuracy
    if let Value::Tensor(predictions) = &inf_results[0] {
        let mut correct = 0;
        for i in 0..num_test {
            // Get predicted class
            let mut max_idx = 0;
            let mut max_val = predictions[[i, 0]];
            for j in 1..10 {
                if predictions[[i, j]] > max_val {
                    max_val = predictions[[i, j]];
                    max_idx = j;
                }
            }

            // Get true class
            let true_class = i % 10; // Our synthetic data labels digits 0-9 in order

            if max_idx == true_class {
                correct += 1;
            }
        }

        let accuracy = correct as f32 / num_test as f32 * 100.0;
        println!("Test Accuracy: {:.1}% ({}/{})", accuracy, correct, num_test);
    }

    // Show some predictions
    println!("\nSample predictions:");
    println!("Index | True | Predicted | Correct");
    println!("------|------|-----------|--------");

    if let Value::Tensor(predictions) = &inf_results[0] {
        for i in 0..10.min(num_test) {
            let mut max_idx = 0;
            let mut max_val = predictions[[i, 0]];
            for j in 1..10 {
                if predictions[[i, j]] > max_val {
                    max_val = predictions[[i, j]];
                    max_idx = j;
                }
            }
            let true_class = i % 10;
            let correct = if max_idx == true_class { "✓" } else { "✗" };
            println!("  {:2}  |  {}   |     {}     |   {}", i, true_class, max_idx, correct);
        }
    }

    println!("\n=== Training Complete ===");
}
