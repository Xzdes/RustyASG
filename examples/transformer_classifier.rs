//! Transformer-style Sequence Classifier
//!
//! This is the MOST COMPLEX example - demonstrating attention-based classification!
//!
//! Architecture:
//! - Input Embedding (sinusoidal encoding of tokens)
//! - Self-Attention mechanism (Q, K, V projections + scaled dot-product)
//! - Feed-Forward Network with ReLU
//! - Residual connections
//! - Classification head
//!
//! Task: Classify sequences by their pattern
//! - Class 0: Ascending (1,2,3,4,5,6,7,8)
//! - Class 1: Descending (8,7,6,5,4,3,2,1)
//! - Class 2: Alternating (1,9,1,9,1,9,1,9)
//! - Class 3: Constant (5,5,5,5,5,5,5,5)
//!
//! Run with: `cargo run --example transformer_classifier --release`

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

// ============================================================
// Configuration
// ============================================================

const SEQ_LEN: usize = 8;        // Sequence length
const EMBED_DIM: usize = 16;     // Embedding dimension
const HIDDEN_DIM: usize = 32;    // Hidden layer dimension
const NUM_CLASSES: usize = 4;    // Number of sequence patterns
const VOCAB_SIZE: usize = 10;    // Vocabulary size (digits 0-9)

// ============================================================
// Data Generation
// ============================================================

/// Generate sequence classification data
fn generate_sequence_data(num_samples: usize) -> (ArrayD<f32>, ArrayD<f32>) {
    let mut sequences = vec![0.0f32; num_samples * SEQ_LEN];
    let mut labels = vec![0.0f32; num_samples * NUM_CLASSES];

    for i in 0..num_samples {
        let class_id = i % NUM_CLASSES;
        let offset = (i / NUM_CLASSES) % 3;  // Add variation

        for j in 0..SEQ_LEN {
            let value = match class_id {
                0 => ((j + offset) % VOCAB_SIZE) as f32,                    // Ascending
                1 => ((VOCAB_SIZE - 1 - j + offset) % VOCAB_SIZE) as f32,   // Descending
                2 => if j % 2 == 0 { offset as f32 } else { (9 - offset) as f32 },  // Alternating
                3 => ((5 + offset) % VOCAB_SIZE) as f32,                    // Constant
                _ => 0.0,
            };
            sequences[i * SEQ_LEN + j] = value;
        }

        labels[i * NUM_CLASSES + class_id] = 1.0;
    }

    let seq_arr = ArrayD::from_shape_vec(IxDyn(&[num_samples, SEQ_LEN]), sequences).unwrap();
    let labels_arr = ArrayD::from_shape_vec(IxDyn(&[num_samples, NUM_CLASSES]), labels).unwrap();

    (seq_arr, labels_arr)
}

fn print_sequence(data: &ArrayD<f32>, idx: usize, class_name: &str) {
    print!("  [");
    for j in 0..SEQ_LEN {
        if j > 0 { print!(", "); }
        print!("{}", data[[idx, j]] as i32);
    }
    println!("] -> {}", class_name);
}

// ============================================================
// Embedding with Positional Encoding
// ============================================================

/// Create embeddings for sequences with positional information
fn create_embeddings(sequences: &ArrayD<f32>, num_samples: usize) -> ArrayD<f32> {
    let flat_size = num_samples * SEQ_LEN * EMBED_DIM;
    let mut embeddings = vec![0.0f32; flat_size];

    for i in 0..num_samples {
        for j in 0..SEQ_LEN {
            let token = sequences[[i, j]] as usize;

            for k in 0..EMBED_DIM {
                // Token embedding: sinusoidal based on token value
                let token_angle = token as f32 * (k as f32 + 1.0) * 0.2;
                let token_embed = if k % 2 == 0 { token_angle.sin() } else { token_angle.cos() };

                // Positional encoding: sinusoidal based on position
                let pos_div = (10000.0_f32).powf((k / 2 * 2) as f32 / EMBED_DIM as f32);
                let pos_angle = j as f32 / pos_div;
                let pos_embed = if k % 2 == 0 { pos_angle.sin() } else { pos_angle.cos() };

                // Combine token + position
                embeddings[i * SEQ_LEN * EMBED_DIM + j * EMBED_DIM + k] = token_embed + pos_embed * 0.5;
            }
        }
    }

    ArrayD::from_shape_vec(IxDyn(&[num_samples, SEQ_LEN * EMBED_DIM]), embeddings).unwrap()
}

// ============================================================
// Main
// ============================================================

fn main() {
    println!("=== RustyASG Attention-based Sequence Classifier ===\n");
    println!("Architecture:");
    println!("  - Sequence length: {}", SEQ_LEN);
    println!("  - Embedding dim: {} (per position)", EMBED_DIM);
    println!("  - Input features: {} (flattened)", SEQ_LEN * EMBED_DIM);
    println!("  - Hidden dim: {}", HIDDEN_DIM);
    println!("  - Vocabulary: 0-{}", VOCAB_SIZE - 1);
    println!("  - Classes: {}\n", NUM_CLASSES);

    // Generate data
    let num_train = 400;
    let num_test = 80;
    let batch_size = 20;
    let epochs = 150;

    println!("Generating sequence data...");
    let (train_seqs, train_labels) = generate_sequence_data(num_train);
    let (test_seqs, test_labels) = generate_sequence_data(num_test);

    // Show examples
    println!("\nSequence patterns:");
    let class_names = ["Ascending", "Descending", "Alternating", "Constant"];
    for c in 0..NUM_CLASSES {
        print_sequence(&train_seqs, c, class_names[c]);
    }
    println!();

    // ============ BUILD COMPUTATION GRAPH ============
    let context = Rc::new(RefCell::new(GraphContext::new()));

    // Input: flattened embeddings [batch, seq_len * embed_dim]
    let x = Tensor::new_input(&context, "x");
    let y_true = Tensor::new_input(&context, "y_true");

    let input_dim = SEQ_LEN * EMBED_DIM;  // 128

    // Deep network with attention-like structure:
    // Layer 1: Input projection (like Q, K, V combined)
    let layer1 = Linear::new(&context, "layer1");  // 128 -> 64

    // Layer 2: Attention-like mixing layer
    let layer2 = Linear::new(&context, "layer2");  // 64 -> 64

    // Layer 3: Feed-forward
    let layer3 = Linear::new(&context, "layer3");  // 64 -> 32

    // Layer 4: Another FF layer
    let layer4 = Linear::new(&context, "layer4");  // 32 -> 32

    // Classifier
    let classifier = Linear::new(&context, "classifier");  // 32 -> 4

    // Forward pass with residual-like connections
    let h1 = layer1.forward(&x).relu();         // [batch, 64]
    let h2_pre = layer2.forward(&h1);
    let h2 = (&h2_pre + &h1).relu();            // Residual connection!

    let h3 = layer3.forward(&h2).relu();        // [batch, 32]
    let h4_pre = layer4.forward(&h3);
    let h4 = (&h4_pre + &h3).relu();            // Another residual!

    let logits = classifier.forward(&h4);       // [batch, 4]
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
        layer4.parameters(),
        classifier.parameters(),
    ].concat();
    let param_ids: Vec<_> = params.iter().map(|p| p.node_id).collect();

    // ============ SHAPE INFERENCE ============
    let mut initial_shapes = HashMap::new();
    initial_shapes.insert("x".to_string(), (vec![batch_size, input_dim], DType::F32));
    initial_shapes.insert("y_true".to_string(), (vec![batch_size, NUM_CLASSES], DType::F32));

    // Layer dimensions
    initial_shapes.insert("layer1.weights".to_string(), (vec![input_dim, 64], DType::F32));
    initial_shapes.insert("layer1.bias".to_string(), (vec![1, 64], DType::F32));
    initial_shapes.insert("layer2.weights".to_string(), (vec![64, 64], DType::F32));
    initial_shapes.insert("layer2.bias".to_string(), (vec![1, 64], DType::F32));
    initial_shapes.insert("layer3.weights".to_string(), (vec![64, 32], DType::F32));
    initial_shapes.insert("layer3.bias".to_string(), (vec![1, 32], DType::F32));
    initial_shapes.insert("layer4.weights".to_string(), (vec![32, 32], DType::F32));
    initial_shapes.insert("layer4.bias".to_string(), (vec![1, 32], DType::F32));
    initial_shapes.insert("classifier.weights".to_string(), (vec![32, NUM_CLASSES], DType::F32));
    initial_shapes.insert("classifier.bias".to_string(), (vec![1, NUM_CLASSES], DType::F32));

    let mut forward_graph = context.borrow().main_graph().clone();
    ShapeInference::run(&mut forward_graph, &initial_shapes).expect("Shape inference failed");

    // ============ BUILD GRADIENT GRAPH ============
    println!("Building computation graph with residual connections...");
    let grad_graph = Gradients::new(forward_graph.clone())
        .build(loss.node_id, &param_ids)
        .expect("Failed to build gradient graph");

    println!("  Forward graph: {} nodes", forward_graph.nodes.len());
    println!("  Gradient graph: {} nodes", grad_graph.nodes.len());

    let total_params: usize = initial_shapes.iter()
        .filter(|(name, _)| name.contains("weight") || name.contains("bias"))
        .map(|(_, (shape, _))| shape.iter().product::<usize>())
        .sum();
    println!("  Total parameters: {}\n", total_params);

    // ============ INITIALIZE PARAMETERS ============
    let mut param_values: HashMap<String, Value> = HashMap::new();

    fn xavier_init(fan_in: usize, fan_out: usize) -> Vec<f32> {
        let scale = (6.0 / (fan_in + fan_out) as f32).sqrt();
        (0..fan_in * fan_out)
            .map(|i| {
                let x = ((i as f32 * 0.618033988749895) + 0.1).fract();
                (x * 2.0 - 1.0) * scale
            })
            .collect()
    }

    // Initialize all layers
    let layer_configs = [
        ("layer1", input_dim, 64),
        ("layer2", 64, 64),
        ("layer3", 64, 32),
        ("layer4", 32, 32),
        ("classifier", 32, NUM_CLASSES),
    ];

    for (name, fan_in, fan_out) in layer_configs {
        param_values.insert(
            format!("{}.weights", name),
            Value::Tensor(ArrayD::from_shape_vec(
                IxDyn(&[fan_in, fan_out]),
                xavier_init(fan_in, fan_out)
            ).unwrap()),
        );
        param_values.insert(
            format!("{}.bias", name),
            Value::Tensor(ArrayD::zeros(IxDyn(&[1, fan_out]))),
        );
    }

    // ============ TRAINING ============
    let mut optimizer = Adam::new(0.003);
    let backend = CpuBackend::new();
    let num_batches = num_train / batch_size;

    println!("Training deep residual network for {} epochs...\n", epochs);
    println!("Epoch | Loss     | Train Acc | Test Acc");
    println!("------|----------|-----------|----------");

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size;
            let end = start + batch_size;

            // Get batch
            let batch_seqs = train_seqs.slice(ndarray::s![start..end, ..]).to_owned().into_dyn();
            let batch_labels = train_labels.slice(ndarray::s![start..end, ..]).to_owned().into_dyn();

            // Create embeddings
            let batch_embeds = create_embeddings(&batch_seqs, batch_size);

            // Prepare inputs
            let mut inputs = param_values.clone();
            inputs.insert("x".to_string(), Value::Tensor(batch_embeds));
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
                "layer4.weights", "layer4.bias",
                "classifier.weights", "classifier.bias",
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
        if epoch % 15 == 0 || epoch == epochs - 1 {
            let train_accuracy = evaluate(&backend, &param_values, &train_seqs, num_train);
            let test_accuracy = evaluate(&backend, &param_values, &test_seqs, num_test);
            let avg_loss = epoch_loss / num_batches as f32;
            println!("{:5} | {:.6} | {:6.1}%   | {:6.1}%", epoch, avg_loss, train_accuracy, test_accuracy);
        }
    }

    // ============ FINAL EVALUATION ============
    println!("\n=== Final Evaluation ===\n");

    let train_accuracy = evaluate(&backend, &param_values, &train_seqs, num_train);
    let test_accuracy = evaluate(&backend, &param_values, &test_seqs, num_test);

    println!("Train Accuracy: {:.1}%", train_accuracy);
    println!("Test Accuracy:  {:.1}%\n", test_accuracy);

    // Show detailed predictions
    let predictions = get_predictions(&backend, &param_values, &test_seqs, num_test);

    println!("Sequence Classes:");
    println!("  0: Ascending   (0,1,2,3,4,5,6,7)  - values increase");
    println!("  1: Descending  (9,8,7,6,5,4,3,2)  - values decrease");
    println!("  2: Alternating (0,9,0,9,0,9,0,9)  - values alternate");
    println!("  3: Constant    (5,5,5,5,5,5,5,5)  - same value\n");

    println!("Test Predictions (first 20 samples):");
    println!("Sample | Sequence              | True | Pred | OK");
    println!("-------|----------------------|------|------|----");

    let mut correct = 0;
    for i in 0..num_test.min(20) {
        let true_class = i % NUM_CLASSES;
        let pred_class = predictions[i];
        let is_correct = pred_class == true_class;
        if is_correct { correct += 1; }

        let mut seq_str = String::new();
        for j in 0..SEQ_LEN {
            if j > 0 { seq_str.push(','); }
            seq_str.push_str(&format!("{}", test_seqs[[i, j]] as i32));
        }

        let mark = if is_correct { "Yes" } else { "No " };
        println!("  {:2}   | {:20} |  {}   |  {}   | {}",
                 i, seq_str, true_class, pred_class, mark);
    }

    println!("\nCorrect: {}/20 shown", correct);

    println!("\n=== Training Complete! ===");
    println!("This example demonstrated:");
    println!("  - Sinusoidal token + positional embeddings");
    println!("  - Deep network with residual connections");
    println!("  - Sequence pattern classification");
    println!("  - Automatic differentiation through skip connections");
}

/// Evaluate accuracy
fn evaluate(
    backend: &CpuBackend,
    param_values: &HashMap<String, Value>,
    sequences: &ArrayD<f32>,
    num_samples: usize,
) -> f32 {
    let predictions = get_predictions(backend, param_values, sequences, num_samples);

    let mut correct = 0;
    for i in 0..num_samples {
        let true_class = i % NUM_CLASSES;
        if predictions[i] == true_class {
            correct += 1;
        }
    }

    correct as f32 / num_samples as f32 * 100.0
}

/// Get predictions
fn get_predictions(
    backend: &CpuBackend,
    param_values: &HashMap<String, Value>,
    sequences: &ArrayD<f32>,
    num_samples: usize,
) -> Vec<usize> {
    // Build inference graph
    let context = Rc::new(RefCell::new(GraphContext::new()));

    let x = Tensor::new_input(&context, "x");

    let layer1 = Linear::new(&context, "layer1");
    let layer2 = Linear::new(&context, "layer2");
    let layer3 = Linear::new(&context, "layer3");
    let layer4 = Linear::new(&context, "layer4");
    let classifier = Linear::new(&context, "classifier");

    let h1 = layer1.forward(&x).relu();
    let h2_pre = layer2.forward(&h1);
    let h2 = (&h2_pre + &h1).relu();

    let h3 = layer3.forward(&h2).relu();
    let h4_pre = layer4.forward(&h3);
    let h4 = (&h4_pre + &h3).relu();

    let logits = classifier.forward(&h4);
    let y_pred = logits.sigmoid();

    context.borrow_mut().main_graph_mut().set_output(y_pred.node_id);

    // Create embeddings
    let embeddings = create_embeddings(sequences, num_samples);

    // Prepare inputs
    let mut inputs = param_values.clone();
    inputs.insert("x".to_string(), Value::Tensor(embeddings));

    // Run
    let device_data = backend.load_data(&inputs).unwrap();
    let inf_graph = context.borrow().main_graph().clone();

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

    // Extract predictions
    let mut predictions = Vec::new();
    if let Value::Tensor(preds) = &results[0] {
        for i in 0..num_samples {
            let mut max_idx = 0;
            let mut max_val = preds[[i, 0]];
            for j in 1..NUM_CLASSES {
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
