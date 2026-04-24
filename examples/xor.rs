//! XOR example — trains a tiny 2-layer MLP to learn the XOR function.
//!
//! Showcases the declarative v0.3 API:
//!   - `Linear::new(ctx, name, in, out)` auto-registers weight shapes.
//!   - `GraphContext::init_parameters()` auto-samples weights from their
//!     initializers (Xavier for weights, Zeros for bias).
//!   - `ShapeInference::run_with_context()` pulls parameter shapes from the
//!     registry — user only supplies input shapes.
//!
//! Run with: `cargo run --release --example xor`

use ndarray::array;
use rustyasg::analysis::shape_inference::ShapeInference;
use rustyasg::asg::{DType, NodeType, Value};
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

    let x_data = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]].into_dyn();
    let y_data = array![[0.0], [1.0], [1.0], [0.0]].into_dyn();

    // ---------- Build graph ----------
    let context = Rc::new(RefCell::new(GraphContext::new()));

    let x = Tensor::new_input(&context, "x");
    let y_true = Tensor::new_input(&context, "y_true");

    // 2 -> 8 -> 1 MLP. Shapes and initializers registered automatically.
    let layer1 = Linear::new(&context, "layer1", 2, 8);
    let layer2 = Linear::new(&context, "layer2", 8, 1);

    let hidden = layer1.forward(&x).relu();
    let y_pred = layer2.forward(&hidden).sigmoid();
    let loss = mse_loss(&y_pred, &y_true);

    context
        .borrow_mut()
        .main_graph_mut()
        .set_output(loss.node_id);

    // ---------- Shape inference (parameter shapes come from registry) ----------
    let mut input_shapes = HashMap::new();
    input_shapes.insert("x".to_string(), (vec![4, 2], DType::F32));
    input_shapes.insert("y_true".to_string(), (vec![4, 1], DType::F32));

    let mut forward_graph = context.borrow().main_graph().clone();
    ShapeInference::run_with_context(&mut forward_graph, &context.borrow(), &input_shapes)
        .expect("Shape inference failed");

    // ---------- Gradient graph ----------
    let params: Vec<Tensor> = [layer1.parameters(), layer2.parameters()].concat();
    let param_ids: Vec<_> = params.iter().map(|p| p.node_id).collect();
    let param_names: Vec<String> = params
        .iter()
        .map(|p| {
            context
                .borrow()
                .main_graph()
                .get_node(p.node_id)
                .unwrap()
                .name
                .clone()
                .unwrap()
        })
        .collect();

    let grad_graph = Gradients::new(forward_graph.clone())
        .build(loss.node_id, &param_ids)
        .expect("Failed to build gradient graph");

    println!(
        "Forward graph: {} nodes, Gradient graph: {} nodes\n",
        forward_graph.nodes.len(),
        grad_graph.nodes.len()
    );

    // ---------- Auto-initialize parameters via registry ----------
    let mut param_values: HashMap<String, Value> = HashMap::new();
    context.borrow().init_parameters(&mut param_values);

    let mut optimizer = Sgd::new(0.5).with_momentum(0.9);
    let backend = CpuBackend::new();

    // ---------- Training ----------
    let epochs = 1000;
    println!("Starting training for {} epochs...\n", epochs);

    for epoch in 0..epochs {
        let mut inputs = param_values.clone();
        inputs.insert("x".to_string(), Value::Tensor(x_data.clone()));
        inputs.insert("y_true".to_string(), Value::Tensor(y_data.clone()));

        let memo = build_memo(&backend, &inputs, &forward_graph);
        let (forward_results, forward_memo) = backend.run(&forward_graph, memo).unwrap();

        let loss_value = match &backend.retrieve_data(&forward_results).unwrap()[0] {
            Value::Tensor(arr) => arr.iter().sum::<f32>(),
            _ => panic!("Expected tensor"),
        };

        let (grad_results, _) = backend.run(&grad_graph, forward_memo).unwrap();
        let grads = backend.retrieve_data(&grad_results).unwrap();

        let mut gradients: HashMap<String, Value> = HashMap::new();
        for (name, grad) in param_names.iter().zip(grads.into_iter()) {
            gradients.insert(name.clone(), grad);
        }

        optimizer.step(&mut param_values, &gradients);

        if epoch % 100 == 0 || epoch == epochs - 1 {
            println!("Epoch {:4}: Loss = {:.6}", epoch, loss_value);
        }
    }

    // ---------- Final evaluation ----------
    println!("\n=== Final Evaluation ===\n");

    let inference_context = Rc::new(RefCell::new(GraphContext::new()));
    let x_inf = Tensor::new_input(&inference_context, "x");
    let layer1_inf = Linear::new(&inference_context, "layer1", 2, 8);
    let layer2_inf = Linear::new(&inference_context, "layer2", 8, 1);
    let pred_inf = layer2_inf
        .forward(&layer1_inf.forward(&x_inf).relu())
        .sigmoid();
    inference_context
        .borrow_mut()
        .main_graph_mut()
        .set_output(pred_inf.node_id);

    let mut inf_inputs = param_values.clone();
    inf_inputs.insert("x".to_string(), Value::Tensor(x_data.clone()));

    let inf_graph = inference_context.borrow().main_graph().clone();
    let memo = build_memo(&backend, &inf_inputs, &inf_graph);
    let (inf_results, _) = backend.run(&inf_graph, memo).unwrap();
    let inf_values = backend.retrieve_data(&inf_results).unwrap();

    if let Value::Tensor(predictions) = &inf_values[0] {
        let inputs_arr = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let targets = [0.0, 1.0, 1.0, 0.0];

        println!("Input      | Target | Prediction | Rounded");
        println!("-----------|--------|------------|--------");
        for i in 0..4 {
            let pred = predictions[[i, 0]];
            let rounded = if pred > 0.5 { 1.0 } else { 0.0 };
            let correct = if rounded == targets[i] { "OK" } else { "FAIL" };
            println!(
                "[{:.0}, {:.0}]     |   {:.0}    |   {:.4}    |   {:.0}  {}",
                inputs_arr[i][0], inputs_arr[i][1], targets[i], pred, rounded, correct
            );
        }
    }

    println!("\n=== Training Complete ===");
}

/// Helper: build initial memo for a graph by matching `inputs` keys to
/// `Input`/`Parameter` node names.
fn build_memo(
    backend: &CpuBackend,
    inputs: &HashMap<String, Value>,
    graph: &rustyasg::asg::Asg,
) -> HashMap<(usize, usize), Value> {
    let device_data = backend.load_data(inputs).unwrap();
    let mut memo = HashMap::new();
    for (name, value) in device_data {
        if let Some(node) = graph.nodes.values().find(|n| match &n.node_type {
            NodeType::Input { name: n } | NodeType::Parameter { name: n } => n == &name,
            _ => false,
        }) {
            memo.insert((graph.id, node.id), value);
        }
    }
    memo
}
