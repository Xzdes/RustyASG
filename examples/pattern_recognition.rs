//! Simple Neural Network for Pattern Recognition
//!
//! This example demonstrates:
//! - Building a multi-layer perceptron (MLP)
//! - Pattern classification with 4 distinct patterns
//! - Training with Adam optimizer
//! - Evaluation and accuracy metrics
//!
//! Network: 64 -> 32 -> 16 -> 4
//!
//! Run with: `cargo run --example simple_cnn`

use ndarray::{ArrayD, IxDyn};
use rustyasg::analysis::shape_inference::ShapeInference;
use rustyasg::asg::{DType, NodeType, Value};
use rustyasg::autograd::Gradients;
use rustyasg::losses::mse_loss_mean;
use rustyasg::nn::{Linear, Module};
use rustyasg::optimizers::{Adam, Optimizer};
use rustyasg::runtime::{backend::Backend, cpu_backend::CpuBackend};
use rustyasg::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// Generate synthetic image data with simple patterns
/// Each class has a distinct visual pattern
fn generate_pattern_data(num_samples: usize, image_size: usize) -> (ArrayD<f32>, ArrayD<f32>) {
    let num_classes = 4;
    let pixels = image_size * image_size;

    let mut images = vec![0.0f32; num_samples * pixels];
    let mut labels = vec![0.0f32; num_samples * num_classes];

    for i in 0..num_samples {
        let class_id = i % num_classes;

        for row in 0..image_size {
            for col in 0..image_size {
                let idx = i * pixels + row * image_size + col;

                // Add some noise based on sample index
                let noise = ((i * 7 + row * 3 + col) % 10) as f32 * 0.02;

                let value = match class_id {
                    0 => {
                        // Horizontal stripes
                        if row % 2 == 0 { 0.9 + noise } else { 0.1 - noise }
                    }
                    1 => {
                        // Vertical stripes
                        if col % 2 == 0 { 0.9 + noise } else { 0.1 - noise }
                    }
                    2 => {
                        // Diagonal pattern (top-left to bottom-right)
                        if (row + col) % 2 == 0 { 0.9 + noise } else { 0.1 - noise }
                    }
                    3 => {
                        // Checkerboard (opposite of diagonal)
                        if (row + col + 1) % 2 == 0 { 0.9 + noise } else { 0.1 - noise }
                    }
                    _ => 0.0,
                };

                images[idx] = value.clamp(0.0, 1.0);
            }
        }

        // One-hot encode label
        labels[i * num_classes + class_id] = 1.0;
    }

    // Flatten to [N, pixels]
    let images_arr = ArrayD::from_shape_vec(
        IxDyn(&[num_samples, pixels]),
        images
    ).unwrap();
    let labels_arr = ArrayD::from_shape_vec(
        IxDyn(&[num_samples, num_classes]),
        labels
    ).unwrap();

    (images_arr, labels_arr)
}

/// Print a sample pattern to console
fn print_pattern(data: &ArrayD<f32>, sample_idx: usize, size: usize) {
    let pixels = size * size;
    println!("Pattern visualization (8x8):");
    for row in 0..size {
        print!("  ");
        for col in 0..size {
            let idx = sample_idx * pixels + row * size + col;
            let val = data.as_slice().unwrap()[idx];
            if val > 0.5 {
                print!("##");
            } else {
                print!("..");
            }
        }
        println!();
    }
}

fn main() {
    println!("=== RustyASG Pattern Recognition Neural Network ===\n");

    // Configuration
    let image_size = 8;  // 8x8 images = 64 input features
    let num_classes = 4;
    let num_train = 200;
    let num_test = 40;
    let batch_size = 20;
    let epochs = 100;
    let input_size = image_size * image_size; // 64

    // Generate data
    println!("Generating pattern data...");
    println!("  - Image size: {}x{} ({} features)", image_size, image_size, input_size);
    println!("  - Classes: {}", num_classes);
    println!("  - Train samples: {}, Test samples: {}\n", num_train, num_test);

    let (train_images, train_labels) = generate_pattern_data(num_train, image_size);
    let (test_images, test_labels) = generate_pattern_data(num_test, image_size);

    // Show sample patterns
    println!("Class 0 (Horizontal stripes):");
    print_pattern(&train_images, 0, image_size);
    println!("\nClass 1 (Vertical stripes):");
    print_pattern(&train_images, 1, image_size);
    println!("\nClass 2 (Diagonal):");
    print_pattern(&train_images, 2, image_size);
    println!("\nClass 3 (Inverse diagonal):");
    print_pattern(&train_images, 3, image_size);
    println!();

    // Create graph context
    let context = Rc::new(RefCell::new(GraphContext::new()));

    // Define inputs
    let x = Tensor::new_input(&context, "x");          // [N, 64]
    let y_true = Tensor::new_input(&context, "y_true"); // [N, 4]

    // Build MLP: 64 -> 32 -> 16 -> 4
    let layer1 = Linear::new(&context, "layer1"); // 64 -> 32
    let layer2 = Linear::new(&context, "layer2"); // 32 -> 16
    let layer3 = Linear::new(&context, "layer3"); // 16 -> 4

    // Forward pass with ReLU activations
    let h1 = layer1.forward(&x).relu();
    let h2 = layer2.forward(&h1).relu();
    let logits = layer3.forward(&h2);
    let y_pred = logits.sigmoid();

    // Loss
    let loss = mse_loss_mean(&y_pred, &y_true);

    // Set output
    context.borrow_mut().main_graph_mut().set_output(loss.node_id);

    // Collect parameters
    let params: Vec<Tensor> = [
        layer1.parameters(),
        layer2.parameters(),
        layer3.parameters(),
    ].concat();
    let param_ids: Vec<_> = params.iter().map(|p| p.node_id).collect();

    // Shape inference
    let mut initial_shapes = HashMap::new();
    initial_shapes.insert("x".to_string(), (vec![batch_size, input_size], DType::F32));
    initial_shapes.insert("y_true".to_string(), (vec![batch_size, num_classes], DType::F32));

    // Layer params
    initial_shapes.insert("layer1.weights".to_string(), (vec![input_size, 32], DType::F32));
    initial_shapes.insert("layer1.bias".to_string(), (vec![1, 32], DType::F32));
    initial_shapes.insert("layer2.weights".to_string(), (vec![32, 16], DType::F32));
    initial_shapes.insert("layer2.bias".to_string(), (vec![1, 16], DType::F32));
    initial_shapes.insert("layer3.weights".to_string(), (vec![16, num_classes], DType::F32));
    initial_shapes.insert("layer3.bias".to_string(), (vec![1, num_classes], DType::F32));

    let mut forward_graph = context.borrow().main_graph().clone();
    ShapeInference::run(&mut forward_graph, &initial_shapes).expect("Shape inference failed");

    // Build gradient graph
    println!("Building computation graph...");
    let grad_graph = Gradients::new(forward_graph.clone())
        .build(loss.node_id, &param_ids)
        .expect("Failed to build gradient graph");

    println!("  Forward graph: {} nodes", forward_graph.nodes.len());
    println!("  Gradient graph: {} nodes\n", grad_graph.nodes.len());

    // Initialize parameters
    let mut param_values: HashMap<String, Value> = HashMap::new();

    // Xavier initialization
    fn xavier_init(fan_in: usize, fan_out: usize) -> Vec<f32> {
        let scale = (6.0 / (fan_in + fan_out) as f32).sqrt();
        (0..fan_in * fan_out)
            .map(|i| {
                // Pseudo-random using golden ratio
                let x = (i as f32 * 0.618033988749895) % 1.0;
                (x * 2.0 - 1.0) * scale
            })
            .collect()
    }

    // Layer 1: 64 -> 32
    param_values.insert(
        "layer1.weights".to_string(),
        Value::Tensor(ArrayD::from_shape_vec(IxDyn(&[input_size, 32]), xavier_init(input_size, 32)).unwrap()),
    );
    param_values.insert(
        "layer1.bias".to_string(),
        Value::Tensor(ArrayD::zeros(IxDyn(&[1, 32]))),
    );

    // Layer 2: 32 -> 16
    param_values.insert(
        "layer2.weights".to_string(),
        Value::Tensor(ArrayD::from_shape_vec(IxDyn(&[32, 16]), xavier_init(32, 16)).unwrap()),
    );
    param_values.insert(
        "layer2.bias".to_string(),
        Value::Tensor(ArrayD::zeros(IxDyn(&[1, 16]))),
    );

    // Layer 3: 16 -> 4
    param_values.insert(
        "layer3.weights".to_string(),
        Value::Tensor(ArrayD::from_shape_vec(IxDyn(&[16, num_classes]), xavier_init(16, num_classes)).unwrap()),
    );
    param_values.insert(
        "layer3.bias".to_string(),
        Value::Tensor(ArrayD::zeros(IxDyn(&[1, num_classes]))),
    );

    // Setup training
    let mut optimizer = Adam::new(0.01);
    let backend = CpuBackend::new();
    let num_batches = num_train / batch_size;

    println!("Training MLP for {} epochs...\n", epochs);
    println!("Epoch | Loss     | Train Acc");
    println!("------|----------|----------");

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = start + batch_size;

            // Get batch
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
                            NodeType::Input { name: n } |
                            NodeType::Parameter { name: n } if n == &name
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
                _ => 0.0,
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

        // Evaluate periodically
        if epoch % 10 == 0 || epoch == epochs - 1 {
            let accuracy = evaluate_accuracy(&backend, &param_values, &train_images, &train_labels, num_train, num_classes);
            let avg_loss = epoch_loss / num_batches as f32;
            println!("{:5} | {:.6} | {:.1}%", epoch, avg_loss, accuracy);
        }
    }

    // Final evaluation on test set
    println!("\n=== Test Set Evaluation ===\n");

    let test_accuracy = evaluate_accuracy(&backend, &param_values, &test_images, &test_labels, num_test, num_classes);
    let train_accuracy = evaluate_accuracy(&backend, &param_values, &train_images, &train_labels, num_train, num_classes);

    println!("Train Accuracy: {:.1}%", train_accuracy);
    println!("Test Accuracy:  {:.1}%\n", test_accuracy);

    // Show predictions
    let predictions = get_predictions(&backend, &param_values, &test_images, num_test, num_classes);

    println!("Pattern Classes:");
    println!("  0: Horizontal stripes  ##..##..##..");
    println!("  1: Vertical stripes    ##########..");
    println!("  2: Diagonal pattern    ##..##..##..");
    println!("  3: Inverse diagonal    ..##..##..##\n");

    println!("Test Set Predictions:");
    println!("Sample | True | Pred | Correct");
    println!("-------|------|------|--------");

    let mut correct_count = 0;
    for i in 0..num_test.min(20) {
        let true_class = i % num_classes;
        let pred_class = predictions[i];
        let is_correct = pred_class == true_class;
        if is_correct { correct_count += 1; }
        let mark = if is_correct { "Yes" } else { "No " };
        println!("  {:3}  |  {}   |  {}   |  {}", i, true_class, pred_class, mark);
    }

    println!("\nTotal: {}/{} correct ({:.1}%)", correct_count, num_test.min(20),
             correct_count as f32 / num_test.min(20) as f32 * 100.0);

    println!("\n=== Training Complete ===");
}

/// Evaluate accuracy on given data
fn evaluate_accuracy(
    backend: &CpuBackend,
    param_values: &HashMap<String, Value>,
    images: &ArrayD<f32>,
    labels: &ArrayD<f32>,
    num_samples: usize,
    num_classes: usize,
) -> f32 {
    let predictions = get_predictions(backend, param_values, images, num_samples, num_classes);

    let mut correct = 0;
    for i in 0..num_samples {
        let true_class = i % num_classes;
        if predictions[i] == true_class {
            correct += 1;
        }
    }

    correct as f32 / num_samples as f32 * 100.0
}

/// Get predictions for given data
fn get_predictions(
    backend: &CpuBackend,
    param_values: &HashMap<String, Value>,
    images: &ArrayD<f32>,
    num_samples: usize,
    num_classes: usize,
) -> Vec<usize> {
    // Create inference graph
    let inf_context = Rc::new(RefCell::new(GraphContext::new()));

    let x_inf = Tensor::new_input(&inf_context, "x");
    let layer1_inf = Linear::new(&inf_context, "layer1");
    let layer2_inf = Linear::new(&inf_context, "layer2");
    let layer3_inf = Linear::new(&inf_context, "layer3");

    let h1 = layer1_inf.forward(&x_inf).relu();
    let h2 = layer2_inf.forward(&h1).relu();
    let logits = layer3_inf.forward(&h2);
    let y_pred = logits.sigmoid();

    inf_context.borrow_mut().main_graph_mut().set_output(y_pred.node_id);

    // Prepare inputs
    let mut inputs = param_values.clone();
    inputs.insert("x".to_string(), Value::Tensor(images.clone()));

    // Load and run
    let device_data = backend.load_data(&inputs).unwrap();
    let inf_graph = inf_context.borrow().main_graph().clone();

    let mut memo = HashMap::new();
    for (name, value) in device_data {
        let node_id = inf_graph
            .nodes
            .iter()
            .find(|(_, node)| {
                matches!(
                    &node.node_type,
                    NodeType::Input { name: n } |
                    NodeType::Parameter { name: n } if n == &name
                )
            })
            .map(|(id, _)| *id);

        if let Some(id) = node_id {
            memo.insert((inf_graph.id, id), value);
        }
    }

    let (results, _) = backend.run(&inf_graph, memo).unwrap();

    // Get predictions
    let mut predictions = Vec::new();

    if let Value::Tensor(preds) = &results[0] {
        for i in 0..num_samples {
            let mut max_idx = 0;
            let mut max_val = preds[[i, 0]];
            for j in 1..num_classes {
                if preds[[i, j]] > max_val {
                    max_val = preds[[i, j]];
                    max_idx = j;
                }
            }
            predictions.push(max_idx);
        }
    }

    predictions
}
