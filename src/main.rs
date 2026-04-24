//! RustyASG entry point: TransformerBlock training demo with the declarative
//! shape/init API.
//!
//! Compared to v0.2 this binary no longer enumerates parameter shapes by
//! string matching — shapes come from `GraphContext::build_shape_map()` and
//! initial weights from `GraphContext::init_parameters()`.

mod gui_viewer;

use rustyasg::analysis::shape_inference::ShapeInference;
use rustyasg::asg::{self, DType, NodeType, Value};
use rustyasg::autograd::Gradients;
use rustyasg::losses::mse_loss;
use rustyasg::nn::{Module, TransformerBlock};
use rustyasg::optimizers::{Optimizer, Sgd};
use rustyasg::runtime::backend::{Backend, Memo};
use rustyasg::runtime::cpu_backend::CpuBackend;
use rustyasg::runtime::wgpu_backend::WgpuBackend;
use rustyasg::tensor::{GraphContext, Tensor};

use crate::gui_viewer::GraphViewerApp;

use clap::Parser;
use eframe::egui;
use ndarray::ArrayD;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about = "RustyASG: deep learning framework", long_about = None)]
struct Args {
    /// Enable interactive graph visualization (GUI).
    #[arg(short, long)]
    visualize: bool,

    /// Use GPU (wgpu) backend. Default: CPU.
    #[arg(long)]
    gpu: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.visualize {
        let (tx, rx) = mpsc::channel();
        let use_gpu = args.gpu;
        thread::spawn(move || {
            println!("[COMPUTATION] Starting computation thread...");
            if let Err(e) = run_computation(Some(tx), use_gpu) {
                eprintln!("[COMPUTATION] Error: {}", e);
            }
        });
        println!("[GUI] Starting GUI on main thread...");
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 720.0]),
            ..Default::default()
        };
        eframe::run_native(
            "RustyASG — Graph Visualizer",
            options,
            Box::new(|cc| Ok(Box::new(GraphViewerApp::new(cc, rx)))),
        )?;
    } else {
        run_computation(None, args.gpu)?;
    }
    Ok(())
}

/// Builds a TransformerBlock, computes its gradient graph, and trains it.
fn run_computation(
    tx: Option<mpsc::Sender<asg::Asg>>,
    use_gpu: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    // ---------- 1. Model hyper-parameters ----------
    let embed_dim = 4;
    let ff_hidden_dim = embed_dim * 4;
    let num_heads = 2;
    let batch_size = 1;

    println!("--- RustyASG full training-loop demo ---");
    println!(
        "[Config] Backend: {}",
        if use_gpu { "GPU (wgpu)" } else { "CPU" }
    );

    // ---------- 2. Build the graph ----------
    let context = Rc::new(RefCell::new(GraphContext::new()));
    let model_input = Tensor::new_input(&context, "input_data");
    let true_output = Tensor::new_input(&context, "true_output");

    // TransformerBlock auto-registers every parameter's shape & initializer.
    let model = TransformerBlock::new(&context, embed_dim, num_heads, ff_hidden_dim, "transformer");
    let model_output = model.forward(&model_input);
    let loss = mse_loss(&model_output, &true_output);

    let mut forward_graph = context.borrow().main_graph().clone();
    forward_graph.set_output(loss.node_id);
    println!("\n[1] Forward graph built.");

    // ---------- 3. Shape inference ----------
    // Only input shapes are user-supplied; parameter shapes come from the registry.
    let mut input_shapes = HashMap::new();
    input_shapes.insert(
        "input_data".to_string(),
        (vec![batch_size, embed_dim], DType::F32),
    );
    input_shapes.insert(
        "true_output".to_string(),
        (vec![batch_size, embed_dim], DType::F32),
    );

    ShapeInference::run_with_context(&mut forward_graph, &context.borrow(), &input_shapes)?;
    println!("[2] Shape inference complete (parameter shapes auto-registered).");

    // ---------- 4. Build gradient graph ----------
    let param_tensors = model.parameters();
    let param_ids: Vec<_> = param_tensors.iter().map(|p| p.node_id).collect();

    let grad_generator = Gradients::new(forward_graph.clone());
    let grad_graph = grad_generator.build(loss.node_id, &param_ids)?;
    println!("[3] Gradient graph built and analyzed.");

    // ---------- 5. Send forward graph to GUI (if enabled) ----------
    if let Some(tx) = tx {
        println!("\n[+] Sending forward graph to visualizer...");
        tx.send(forward_graph.clone())?;
    }

    // ---------- 6. Initial runtime data ----------
    let mut runtime_data: HashMap<String, Value> = HashMap::new();
    runtime_data.insert(
        "input_data".to_string(),
        Value::Tensor(ArrayD::random(
            ndarray::IxDyn(&[batch_size, embed_dim]),
            Uniform::new(-1.0, 1.0),
        )),
    );
    runtime_data.insert(
        "true_output".to_string(),
        Value::Tensor(ArrayD::from_elem(
            ndarray::IxDyn(&[batch_size, embed_dim]),
            0.5,
        )),
    );

    // Auto-initialize every registered parameter with its declared initializer.
    context.borrow().init_parameters(&mut runtime_data);

    let optimizer = Sgd::new(0.01);
    println!("[4] Runtime data and optimizer initialized (weights auto-sampled).");

    // ---------- 7. Pick backend and train ----------
    if use_gpu {
        let backend = pollster::block_on(WgpuBackend::new());
        run_training_loop(
            backend,
            forward_graph,
            grad_graph,
            runtime_data,
            param_tensors,
            optimizer,
        );
    } else {
        let backend = CpuBackend::new();
        run_training_loop(
            backend,
            forward_graph,
            grad_graph,
            runtime_data,
            param_tensors,
            optimizer,
        );
    }

    Ok(())
}

fn run_training_loop<B: Backend>(
    backend: B,
    forward_graph: asg::Asg,
    grad_graph: asg::Asg,
    mut runtime_data: HashMap<String, Value>,
    param_tensors: Vec<Tensor>,
    mut optimizer: Sgd,
) {
    let param_names: Vec<String> = param_tensors
        .iter()
        .map(|p| {
            let ctx = p.context.borrow();
            let graph = ctx.main_graph();
            graph
                .get_node(p.node_id)
                .expect("parameter tensor references a node that was not added to the graph")
                .name
                .as_ref()
                .expect("parameter nodes always have a name")
                .clone()
        })
        .collect();

    println!("\n--- TRAINING LOOP ---\n");
    let start_time = Instant::now();

    for epoch in 0..15 {
        // 7.1. Load data to device.
        let device_data = backend.load_data(&runtime_data).unwrap();
        let mut initial_memo: Memo<B::DeviceData> = HashMap::new();
        for (name, data) in device_data {
            if let Some(node) = forward_graph.nodes.values().find(|n| match &n.node_type {
                NodeType::Input { name: n_name } | NodeType::Parameter { name: n_name } => {
                    n_name == &name
                }
                _ => false,
            }) {
                initial_memo.insert((forward_graph.id, node.id), data);
            }
        }

        // 7.2. Forward pass.
        let (loss_device_vec, forward_memo) = backend.run(&forward_graph, initial_memo).unwrap();
        let loss_value_vec = backend.retrieve_data(&loss_device_vec).unwrap();
        let loss_value = loss_value_vec.first().unwrap();

        // 7.3. Backward pass.
        let (grad_device_vec, _) = backend.run(&grad_graph, forward_memo).unwrap();
        let grad_value_vec = backend.retrieve_data(&grad_device_vec).unwrap();

        // 7.4. Gather gradients by name.
        let mut computed_grads = HashMap::new();
        for (name, value) in param_names.iter().zip(grad_value_vec.into_iter()) {
            computed_grads.insert(name.clone(), value);
        }

        // 7.5. Optimizer step.
        optimizer.step(&mut runtime_data, &computed_grads);

        // 7.6. Log loss.
        if let Value::Tensor(loss_tensor) = loss_value {
            println!(
                "Epoch: {:<2}, Loss: {:.6}",
                epoch + 1,
                loss_tensor.first().unwrap_or(&-1.0)
            );
        }
    }

    println!(
        "\n--- TRAINING COMPLETE in {:.2?} ---",
        start_time.elapsed()
    );
}
