//  src/main.rs  (полностью новый)
//! Главный исполняемый файл с корректной инициализацией параметров.

mod analysis;
mod asg;
mod autograd;
mod gui_viewer;
mod losses;
mod nn;
mod optimizers;
mod runtime;
mod tensor;

use crate::analysis::shape_inference::ShapeInference;
use crate::asg::{DType, NodeType, Shape, Value};
use crate::autograd::Gradients;
use crate::gui_viewer::GraphViewerApp;
use crate::losses::mse_loss;
use crate::nn::{Module, TransformerBlock};
use crate::optimizers::{Optimizer, Sgd};
use crate::runtime::backend::{Backend, Memo};
use crate::runtime::cpu_backend::CpuBackend;
use crate::runtime::wgpu_backend::WgpuBackend;
use crate::tensor::{GraphContext, Tensor};

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

/// Аргументы командной строки
#[derive(Parser, Debug)]
#[command(author, version, about = "RustyASG: deep learning framework", long_about = None)]
struct Args {
    /// Включить GUI-визуализацию графа
    #[arg(short, long)]
    visualize: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.visualize {
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            println!("[COMPUTATION] Запуск вычислительного потока...");
            if let Err(e) = run_computation(Some(tx)) {
                eprintln!("[COMPUTATION] Ошибка: {}", e);
            }
        });
        println!("[GUI] Запуск GUI в основном потоке...");
        let options = eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 720.0]),
            ..Default::default()
        };
        eframe::run_native(
            "RustyASG - Визуализатор Графа",
            options,
            Box::new(|cc| Ok(Box::new(GraphViewerApp::new(cc, rx)))),
        )?;
    } else {
        run_computation(None)?;
    }
    Ok(())
}

/// Вся логика построения графа и обучения
fn run_computation(tx: Option<mpsc::Sender<asg::Asg>>) -> Result<(), Box<dyn std::error::Error>> {
    // ---------- 1. Конфигурация ----------
    let use_gpu = true; // <--- ПОМЕНЯЙТЕ НА `false` ДЛЯ ЗАПУСКА НА CPU
    let embed_dim = 4;
    let ff_hidden_dim = embed_dim * 4;
    let num_heads = 2;

    println!("--- Демонстрация полного цикла обучения RustyASG ---");
    println!("[Config] Backend: {}", if use_gpu { "GPU (wgpu)" } else { "CPU" });

    // ---------- 2. Построение графа ----------
    let context = Rc::new(RefCell::new(GraphContext::new()));
    let model_input = Tensor::new_input(&context, "input_data");
    let true_output = Tensor::new_input(&context, "true_output");

    let model = TransformerBlock::new(&context, embed_dim, num_heads, ff_hidden_dim, "transformer");
    let model_output = model.forward(&model_input);
    let loss = mse_loss(&model_output, &true_output);

    let mut forward_graph = context.borrow().main_graph().clone();
    forward_graph.set_output(loss.node_id);
    println!("\n[1] Граф прямого прохода успешно построен.");

    // ---------- 3. Анализ форм ----------
    let mut initial_shapes = HashMap::new();
    initial_shapes.insert("input_data".to_string(), (vec![1, embed_dim], DType::F32));
    initial_shapes.insert("true_output".to_string(), (vec![1, embed_dim], DType::F32));

    for param in model.parameters() {
        let name = context
            .borrow()
            .main_graph()
            .get_node(param.node_id)
            .unwrap()
            .name
            .as_ref()
            .unwrap()
            .clone();
        let shape: Shape = if name.contains("w_q.weights")
            || name.contains("w_k.weights")
            || name.contains("w_v.weights")
            || name.contains("w_o.weights")
        {
            vec![embed_dim, embed_dim]
        } else if name.contains("linear1.weights") {
            vec![embed_dim, ff_hidden_dim]
        } else if name.contains("linear2.weights") {
            vec![ff_hidden_dim, embed_dim]
        } else if name.contains("linear1.bias") {
            vec![1, ff_hidden_dim]
        } else if name.contains("linear2.bias") {
            vec![1, embed_dim]
        } else if name.contains("gamma") {
            vec![1, embed_dim]
        } else if name.contains("beta") {
            vec![1, embed_dim]
        } else {
            vec![1, embed_dim]
        };
        initial_shapes.insert(name, (shape, DType::F32));
    }

    ShapeInference::run(&mut forward_graph, &initial_shapes)?;
    println!("[2] Анализ форм (Shape Inference) для прямого графа завершен.");

    // ---------- 4. Построение графа градиентов ----------
    let param_tensors = model.parameters();
    let param_ids: Vec<_> = param_tensors.iter().map(|p| p.node_id).collect();
    println!("[DEBUG] Параметры, по которым считаются градиенты: {:?}", param_ids);

    let grad_generator = Gradients::new(forward_graph.clone());
    let mut grad_graph = grad_generator.build(loss.node_id, &param_ids)?;
    println!("[3] Граф градиентов построен и проанализирован.");

    // ---------- 5. Отправка графа в GUI (если нужно) ----------
    if let Some(tx) = tx {
        println!("\n[+] Отправка графа прямого прохода в окно визуализации...");
        tx.send(forward_graph.clone())?;
    }

    // ---------- 6. Инициализация данных и оптимизатора ----------
    let mut runtime_data = HashMap::new();
    runtime_data.insert(
        "input_data".to_string(),
        Value::Tensor(ArrayD::random(
            ndarray::IxDyn(&[1, embed_dim]),
            Uniform::new(-1.0, 1.0),
        )),
    );
    runtime_data.insert(
        "true_output".to_string(),
        Value::Tensor(ArrayD::from_elem(ndarray::IxDyn(&[1, embed_dim]), 0.5)),
    );

    // **КЛЮЧЕВОЕ:** инициализируем ТОЛЬКО Linear-веса и bias.
    // gamma/beta остаются обучаемыми параметрами — не трогаем их здесь!
    for (name, (shape, _)) in initial_shapes.iter() {
        if name.contains("input") || name.contains("output") {
            continue;
        }
        if name.contains("gamma") || name.contains("beta") {
            // Обучаемые параметры LayerNorm — случайная инициализация около 1.0 / 0.0
            let init = if name.contains("gamma") {
                ArrayD::random(shape.clone(), Uniform::new(0.9, 1.1))
            } else {
                ArrayD::random(shape.clone(), Uniform::new(-0.1, 0.1))
            };
            runtime_data.insert(name.clone(), Value::Tensor(init));
            continue;
        }
        if name.contains("bias") {
            runtime_data.insert(name.clone(), Value::Tensor(ArrayD::zeros(shape.clone())));
            continue;
        }
        // Остальные веса (линейки) — случайно
        runtime_data.insert(
            name.clone(),
            Value::Tensor(ArrayD::random(shape.clone(), Uniform::new(-0.1, 0.1))),
        );
    }

    let optimizer = Sgd::new(0.01);
    println!("[4] Данные, веса и оптимизатор инициализированы.");

    // ---------- 7. Выбор бэкенда и запуск обучения ----------
    if use_gpu {
        let backend = pollster::block_on(WgpuBackend::new());
        run_training_loop(backend, forward_graph, grad_graph, runtime_data, param_tensors, optimizer);
    } else {
        let backend = CpuBackend::new();
        run_training_loop(backend, forward_graph, grad_graph, runtime_data, param_tensors, optimizer);
    }

    Ok(())
}

/// Универсальный цикл обучения
fn run_training_loop<B: Backend>(
    backend: B,
    forward_graph: asg::Asg,
    grad_graph: asg::Asg,
    mut runtime_data: HashMap<String, Value>,
    param_tensors: Vec<Tensor>,
    optimizer: Sgd,
) {
    let param_names: Vec<String> = param_tensors
        .iter()
        .map(|p| {
            let ctx = p.context.borrow();
            let graph = ctx.main_graph();
            graph.get_node(p.node_id).unwrap().name.as_ref().unwrap().clone()
        })
        .collect();

    println!("\n--- НАЧАЛО ЦИКЛА ОБУЧЕНИЯ ---\n");
    let start_time = Instant::now();

    for epoch in 0..15 {
        // ---------- 7.1. Загрузка данных на устройство ----------
        let device_data = backend.load_data(&runtime_data).unwrap();
        let mut initial_memo: Memo<B::DeviceData> = HashMap::new();
        for (name, data) in device_data {
            if let Some(node) = forward_graph.nodes.values().find(|n| match &n.node_type {
                NodeType::Input { name: n_name } | NodeType::Parameter { name: n_name } => n_name == &name,
                _ => false,
            }) {
                initial_memo.insert((forward_graph.id, node.id), data);
            }
        }

        // ---------- 7.2. Прямой проход ----------
        let (loss_device_vec, forward_memo) = backend.run(&forward_graph, initial_memo).unwrap();
        let loss_value_vec = backend.retrieve_data(&loss_device_vec).unwrap();
        let loss_value = loss_value_vec.first().unwrap();

        // ---------- 7.3. Обратный проход ----------
        let (grad_device_vec, _) = backend.run(&grad_graph, forward_memo).unwrap();
        let grad_value_vec = backend.retrieve_data(&grad_device_vec).unwrap();

        // ---------- 7.4. Сбор градиентов ----------
        let mut computed_grads = HashMap::new();
        for (name, value) in param_names.iter().zip(grad_value_vec.into_iter()) {
            computed_grads.insert(name.clone(), value);
        }

        // ---------- 7.5. Шаг оптимизатора ----------
        optimizer.step(&mut runtime_data, &computed_grads);

        // ---------- 7.6. Лог ----------
        if let Value::Tensor(loss_tensor) = loss_value {
            println!("Эпоха: {:<2}, Потери (Loss): {:.6}", epoch + 1, loss_tensor.first().unwrap_or(&-1.0));
        }
    }

    let duration = start_time.elapsed();
    println!("\n--- ОБУЧЕНИЕ ЗАВЕРШЕНО ЗА {:.2?} ---", duration);
}