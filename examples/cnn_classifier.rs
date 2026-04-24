//! CNN classifier example: small convolutional network on synthetic 2D patterns.
//!
//! Demonstrates the full CNN stack on RustyASG:
//!   Conv2d -> ReLU -> AvgPool2d -> Conv2d -> ReLU -> AdaptiveAvgPool2d
//!   -> reshape (flatten) -> Linear -> Linear (classifier).
//!
//! Uses Conv2d autograd (`Conv2dBackwardInput`/`Conv2dBackwardWeight`) and
//! `AvgUnpool2d` / `AdaptiveAvgPool2d` backward, all of which run on both CPU
//! and GPU. This is the smallest self-contained CNN example that trains to
//! perfect accuracy on a tiny synthetic dataset.
//!
//! Run with: `cargo run --release --example cnn_classifier`

// Demo code: readability-first style.
#![allow(clippy::needless_range_loop)]

use ndarray::{ArrayD, IxDyn};
use rustyasg::analysis::shape_inference::ShapeInference;
use rustyasg::asg::{DType, NodeType, Value};
use rustyasg::autograd::Gradients;
use rustyasg::losses::mse_loss_mean;
use rustyasg::nn::{AdaptiveAvgPool2d, AvgPool2d, Conv2d, Linear, Module};
use rustyasg::optimizers::{Adam, Optimizer};
use rustyasg::runtime::{backend::Backend, cpu_backend::CpuBackend};
use rustyasg::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

// ----------------------- synthetic dataset -----------------------

const NUM_CLASSES: usize = 3;
const IMG_H: usize = 8;
const IMG_W: usize = 8;

/// Generate N samples of 1-channel 8x8 images with three distinct "shapes":
///   class 0: solid dot in the top-left quadrant
///   class 1: solid dot in the top-right quadrant
///   class 2: solid dot in the bottom-center
fn generate_shapes(num_samples: usize) -> (ArrayD<f32>, ArrayD<f32>) {
    // Single-channel images: shape [N, 1, H, W] flattened.
    let mut imgs = vec![0.0_f32; num_samples * IMG_H * IMG_W];
    let mut labels = vec![0.0_f32; num_samples * NUM_CLASSES];

    for i in 0..num_samples {
        let class_id = i % NUM_CLASSES;
        let (cy, cx) = match class_id {
            0 => (2, 2),
            1 => (2, 5),
            2 => (5, 4),
            _ => unreachable!(),
        };
        // Paint a 3x3 blob around (cy, cx).
        for dy in 0..3 {
            for dx in 0..3 {
                let y = cy + dy - 1;
                let x = cx + dx - 1;
                if y < IMG_H && x < IMG_W {
                    let idx = i * IMG_H * IMG_W + y * IMG_W + x;
                    imgs[idx] = 1.0;
                }
            }
        }
        labels[i * NUM_CLASSES + class_id] = 1.0;
    }

    let x = ArrayD::from_shape_vec(IxDyn(&[num_samples, 1, IMG_H, IMG_W]), imgs).unwrap();
    let y = ArrayD::from_shape_vec(IxDyn(&[num_samples, NUM_CLASSES]), labels).unwrap();
    (x, y)
}

fn print_image(x: &ArrayD<f32>, sample_idx: usize) {
    for y in 0..IMG_H {
        print!("  ");
        for xp in 0..IMG_W {
            let v = x[[sample_idx, 0, y, xp]];
            print!("{}", if v > 0.5 { "##" } else { ".." });
        }
        println!();
    }
}

// ----------------------- training -----------------------

fn main() {
    println!("=== RustyASG CNN Classifier ===\n");

    let num_train = 60; // 20 per class
    let num_test = 12;
    let batch_size = num_train; // full-batch — dataset is tiny
    let epochs = 600;

    let (x_train, y_train) = generate_shapes(num_train);
    let (x_test, y_test) = generate_shapes(num_test);

    println!("Training examples, one per class:");
    for c in 0..NUM_CLASSES {
        println!("  Class {}:", c);
        print_image(&x_train, c);
    }
    println!();

    // ---------- Build graph ----------
    let ctx = Rc::new(RefCell::new(GraphContext::new()));
    let x = Tensor::new_input(&ctx, "x"); // [N, 1, 8, 8]
    let y_true = Tensor::new_input(&ctx, "y_true"); // [N, 3]

    // Architecture:
    //   Conv2d(1 -> 8, 3x3, pad=1) -> ReLU -> AvgPool(2x2)  => [N, 8, 4, 4]
    //   Conv2d(8 -> 16, 3x3, pad=1) -> ReLU -> GAP           => [N, 16, 1, 1]
    //   flatten -> Linear(16 -> 8) -> ReLU -> Linear(8 -> 3) -> sigmoid
    let conv1 = Conv2d::new(&ctx, "conv1", 1, 8, (3, 3)).with_padding((1, 1));
    let pool1 = AvgPool2d::new((2, 2), (2, 2));
    let conv2 = Conv2d::new(&ctx, "conv2", 8, 16, (3, 3)).with_padding((1, 1));
    let gap = AdaptiveAvgPool2d::global();
    let fc1 = Linear::new(&ctx, "fc1", 16, 8);
    let fc2 = Linear::new(&ctx, "fc2", 8, NUM_CLASSES);

    // Forward pass.
    let h1 = conv1.forward(&x).relu();
    let h1 = pool1.forward(&h1); // [N, 8, 4, 4]
    let h2 = conv2.forward(&h1).relu(); // [N, 16, 4, 4]
    let h2 = gap.forward(&h2); // [N, 16, 1, 1]
    let flat = h2.reshape(vec![num_train as i64, 16]); // [N, 16]
    let h3 = fc1.forward(&flat).relu(); // [N, 8]
    let logits = fc2.forward(&h3); // [N, 3]
    let y_pred = logits.sigmoid();
    let loss = mse_loss_mean(&y_pred, &y_true);

    ctx.borrow_mut().main_graph_mut().set_output(loss.node_id);

    let params: Vec<Tensor> = [
        conv1.parameters(),
        conv2.parameters(),
        fc1.parameters(),
        fc2.parameters(),
    ]
    .concat();
    let param_ids: Vec<_> = params.iter().map(|p| p.node_id).collect();
    let param_names: Vec<String> = params
        .iter()
        .map(|p| {
            ctx.borrow()
                .main_graph()
                .get_node(p.node_id)
                .unwrap()
                .name
                .clone()
                .unwrap()
        })
        .collect();

    // ---------- Shape inference + gradient graph ----------
    let mut input_shapes = HashMap::new();
    input_shapes.insert("x".into(), (vec![batch_size, 1, IMG_H, IMG_W], DType::F32));
    input_shapes.insert("y_true".into(), (vec![batch_size, NUM_CLASSES], DType::F32));

    let mut forward_graph = ctx.borrow().main_graph().clone();
    ShapeInference::run_with_context(&mut forward_graph, &ctx.borrow(), &input_shapes)
        .expect("shape inference");

    let grad_graph = Gradients::new(forward_graph.clone())
        .build(loss.node_id, &param_ids)
        .expect("gradient build");

    println!(
        "Forward graph: {} nodes, gradient graph: {} nodes\n",
        forward_graph.nodes.len(),
        grad_graph.nodes.len()
    );

    // ---------- Auto-init weights via registry ----------
    let mut runtime: HashMap<String, Value> = HashMap::new();
    ctx.borrow().init_parameters(&mut runtime);

    let mut optimizer = Adam::new(0.005);
    let backend = CpuBackend::new();

    println!("Training for {} epochs...\n", epochs);
    for epoch in 0..epochs {
        let mut inputs = runtime.clone();
        inputs.insert("x".into(), Value::Tensor(x_train.clone()));
        inputs.insert("y_true".into(), Value::Tensor(y_train.clone()));

        let memo = build_memo(&backend, &inputs, &forward_graph);
        let (fwd, fwd_memo) = backend.run(&forward_graph, memo).unwrap();
        let loss_val = match &backend.retrieve_data(&fwd).unwrap()[0] {
            Value::Tensor(t) => *t.first().unwrap(),
            _ => panic!("expected tensor"),
        };

        let (grad_dev, _) = backend.run(&grad_graph, fwd_memo).unwrap();
        let grads = backend.retrieve_data(&grad_dev).unwrap();
        let mut grad_map: HashMap<String, Value> = HashMap::new();
        for (name, g) in param_names.iter().zip(grads.into_iter()) {
            grad_map.insert(name.clone(), g);
        }
        optimizer.step(&mut runtime, &grad_map);

        if epoch % 60 == 0 || epoch == epochs - 1 {
            println!("Epoch {:3}: loss = {:.6}", epoch, loss_val);
        }
    }

    // ---------- Evaluate on test set ----------
    // Rebuild inference graph with test-sized batch.
    let inf_ctx = Rc::new(RefCell::new(GraphContext::new()));
    let x_inf = Tensor::new_input(&inf_ctx, "x");
    let conv1_inf = Conv2d::new(&inf_ctx, "conv1", 1, 8, (3, 3)).with_padding((1, 1));
    let pool1_inf = AvgPool2d::new((2, 2), (2, 2));
    let conv2_inf = Conv2d::new(&inf_ctx, "conv2", 8, 16, (3, 3)).with_padding((1, 1));
    let gap_inf = AdaptiveAvgPool2d::global();
    let fc1_inf = Linear::new(&inf_ctx, "fc1", 16, 8);
    let fc2_inf = Linear::new(&inf_ctx, "fc2", 8, NUM_CLASSES);

    let h1 = conv1_inf.forward(&x_inf).relu();
    let h1 = pool1_inf.forward(&h1);
    let h2 = conv2_inf.forward(&h1).relu();
    let h2 = gap_inf.forward(&h2);
    let flat = h2.reshape(vec![num_test as i64, 16]);
    let h3 = fc1_inf.forward(&flat).relu();
    let logits = fc2_inf.forward(&h3);
    let y_pred_inf = logits.sigmoid();
    inf_ctx
        .borrow_mut()
        .main_graph_mut()
        .set_output(y_pred_inf.node_id);

    let mut inf_shapes = HashMap::new();
    inf_shapes.insert("x".into(), (vec![num_test, 1, IMG_H, IMG_W], DType::F32));
    let mut inf_graph = inf_ctx.borrow().main_graph().clone();
    ShapeInference::run_with_context(&mut inf_graph, &inf_ctx.borrow(), &inf_shapes)
        .expect("shape inference");

    let mut inf_inputs = runtime.clone();
    inf_inputs.insert("x".into(), Value::Tensor(x_test.clone()));
    let memo = build_memo(&backend, &inf_inputs, &inf_graph);
    let (results, _) = backend.run(&inf_graph, memo).unwrap();
    let preds = match &backend.retrieve_data(&results).unwrap()[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!("expected tensor"),
    };

    let mut correct = 0;
    println!("\nTest predictions:");
    println!(" idx | true | pred | correct");
    println!(" ----|------|------|--------");
    for i in 0..num_test {
        let true_class = (0..NUM_CLASSES)
            .max_by(|a, b| y_test[[i, *a]].partial_cmp(&y_test[[i, *b]]).unwrap())
            .unwrap();
        let pred_class = (0..NUM_CLASSES)
            .max_by(|a, b| preds[[i, *a]].partial_cmp(&preds[[i, *b]]).unwrap())
            .unwrap();
        let ok = true_class == pred_class;
        if ok {
            correct += 1;
        }
        println!(
            " {:3} | {}    | {}    | {}",
            i,
            true_class,
            pred_class,
            if ok { "yes" } else { "no" }
        );
    }
    println!(
        "\nAccuracy: {}/{} ({:.1}%)",
        correct,
        num_test,
        100.0 * correct as f32 / num_test as f32
    );
}

fn build_memo(
    backend: &CpuBackend,
    inputs: &HashMap<String, Value>,
    graph: &rustyasg::asg::Asg,
) -> HashMap<(usize, usize), Value> {
    let device = backend.load_data(inputs).unwrap();
    let mut memo = HashMap::new();
    for (name, value) in device {
        if let Some(node) = graph.nodes.values().find(|n| match &n.node_type {
            NodeType::Input { name: n } | NodeType::Parameter { name: n } => n == &name,
            _ => false,
        }) {
            memo.insert((graph.id, node.id), value);
        }
    }
    memo
}
