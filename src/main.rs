// --- Файл: src/main.rs ---

//! Главный исполняемый файл с правильной многопоточной архитектурой для GUI.

// --- 1. Объявление модулей проекта ---
mod analysis;
mod asg;
mod autograd;
mod gui_viewer;
mod losses;
mod nn;
mod optimizers;
mod runtime;
mod tensor;

// --- 2. Импорт зависимостей ---
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

/// Структура для парсинга аргументов командной строки
#[derive(Parser, Debug)]
#[command(author = "Xzdes", version, about = "RustyASG: A deep learning framework in Rust", long_about = None)]
struct Args {
    /// Включить нативную real-time визуализацию графа
    #[arg(short, long)]
    visualize: bool,
}

// --- 3. Главная функция ---
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    if args.visualize {
        // --- ПУТЬ С ВИЗУАЛИЗАЦИЕЙ ---
        let (tx, rx) = mpsc::channel();

        // Запускаем тяжелые вычисления в фоновом потоке.
        // `move` передает владение `tx` в этот поток.
        thread::spawn(move || {
            println!("[COMPUTATION] Запуск вычислительного потока...");
            if let Err(e) = run_computation(Some(tx)) {
                eprintln!("[COMPUTATION] Ошибка в вычислительном потоке: {}", e);
            }
        });

        // Основной поток становится потоком GUI. Это блокирующая операция.
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
        // --- ПУТЬ БЕЗ ВИЗУАЛИЗАЦИИ ---
        // Просто выполняем вычисления в основном потоке, как и раньше.
        run_computation(None)?;
    }

    Ok(())
}

/// Эта функция содержит всю логику построения графа и обучения.
/// Она может быть вызвана как из основного, так и из фонового потока.
fn run_computation(tx: Option<mpsc::Sender<asg::Asg>>) -> Result<(), Box<dyn std::error::Error>> {
    // --- Конфигурация ---
    let use_gpu = true; // <--- ПОМЕНЯЙТЕ НА `false` ДЛЯ ЗАПУСКА НА CPU
    let embed_dim = 4;
    let ff_hidden_dim = embed_dim * 4;
    let num_heads = 2;

    println!("--- Демонстрация полного цикла обучения RustyASG ---");
    println!("[Config] Backend: {}", if use_gpu { "GPU (wgpu)" } else { "CPU" });

    // --- 1. Построение Графа ---
    let context = Rc::new(RefCell::new(GraphContext::new()));
    let model_input = Tensor::new_input(&context, "input_data");
    let true_output = Tensor::new_input(&context, "true_output");

    let model = TransformerBlock::new(&context, embed_dim, num_heads, ff_hidden_dim, "transformer");
    let model_output = model.forward(&model_input);
    let loss = mse_loss(&model_output, &true_output);

    let mut forward_graph = context.borrow().main_graph().clone();
    forward_graph.set_output(loss.node_id);
    println!("\n[1] Граф прямого прохода успешно построен.");

    // --- 2. Анализ Форм ---
    let mut initial_shapes = HashMap::new();
    initial_shapes.insert("input_data".to_string(), (vec![1, embed_dim], DType::F32));
    initial_shapes.insert("true_output".to_string(), (vec![1, embed_dim], DType::F32));
    for param in model.parameters() {
        let name = context.borrow().main_graph().get_node(param.node_id)?.name.clone().unwrap();
        let shape: Shape = if name.contains("w_q.weights") || name.contains("w_k.weights") || name.contains("w_v.weights") || name.contains("w_o.weights") {
            vec![embed_dim, embed_dim]
        } else if name.contains("linear1.weights") {
            vec![embed_dim, ff_hidden_dim]
        } else if name.contains("linear2.weights") {
            vec![ff_hidden_dim, embed_dim]
        } else if name.contains("linear1.bias") {
            vec![1, ff_hidden_dim]
        } else {
            vec![1, embed_dim]
        };
        initial_shapes.insert(name, (shape, DType::F32));
    }

    ShapeInference::run(&mut forward_graph, &initial_shapes)?;
    println!("[2] Анализ форм (Shape Inference) для прямого графа завершен.");
    
    // --- 3. Построение Графа Градиентов ---
    let param_tensors = model.parameters();
    let param_ids: Vec<_> = param_tensors.iter().map(|p| p.node_id).collect();
    let grad_generator = Gradients::new(forward_graph.clone());
    let mut grad_graph = grad_generator.build(loss.node_id, &param_ids)?;
    
    // --- 4. Анализ Форм для Графа Градиентов ---
    let mut grad_initial_shapes = HashMap::new();
    for node in forward_graph.nodes.values() {
        if let Some(shape) = &node.shape {
            let external_node_name = format!("external_{}_{}", forward_graph.id, node.id);
            grad_initial_shapes.insert(external_node_name, (shape.clone(), DType::F32));
        }
    }
    ShapeInference::run(&mut grad_graph, &grad_initial_shapes)?;
    println!("[3] Граф градиентов построен и проанализирован.");

    // --- ОТПРАВКА ГРАФА В GUI-ПОТОК (если он существует) ---
    if let Some(tx) = tx {
        println!("\n[+] Отправка графа прямого прохода в окно визуализации...");
        tx.send(forward_graph.clone())?;
    }

    // --- 5. Инициализация Данных и Оптимизатора ---
    let mut runtime_data = HashMap::new();
    runtime_data.insert("input_data".to_string(), Value::Tensor(ArrayD::random(ndarray::IxDyn(&[1, embed_dim]), Uniform::new(-1.0, 1.0))));
    runtime_data.insert("true_output".to_string(), Value::Tensor(ArrayD::from_elem(ndarray::IxDyn(&[1, embed_dim]), 0.5)));
    for (name, (shape, _)) in initial_shapes.iter() {
        if name.contains("input") || name.contains("output") { continue; }
        if name.contains("gamma") {
            runtime_data.insert(name.clone(), Value::Tensor(ArrayD::ones(shape.clone())));
        } else if name.contains("beta") || name.contains("bias") {
            runtime_data.insert(name.clone(), Value::Tensor(ArrayD::zeros(shape.clone())));
        } else {
            runtime_data.insert(name.clone(), Value::Tensor(ArrayD::random(shape.clone(), Uniform::new(-0.1, 0.1))));
        }
    }
    
    let optimizer = Sgd::new(0.01);
    println!("[4] Данные, веса и оптимизатор инициализированы.");

    // --- 6. Выбор Бэкенда и Запуск Обучения ---
    if use_gpu {
        let backend = pollster::block_on(WgpuBackend::new());
        run_training_loop(backend, forward_graph, grad_graph, runtime_data, param_tensors, optimizer);
    } else {
        let backend = CpuBackend::new();
        run_training_loop(backend, forward_graph, grad_graph, runtime_data, param_tensors, optimizer);
    }
    
    Ok(())
}

/// Универсальный Цикл Обучения
fn run_training_loop<B: Backend>(
    backend: B,
    forward_graph: asg::Asg,
    grad_graph: asg::Asg,
    mut runtime_data: HashMap<String, Value>,
    param_tensors: Vec<Tensor>,
    optimizer: Sgd,
) {
    let param_names: Vec<String> = param_tensors.iter().map(|p| {
        let ctx = p.context.borrow();
        let graph = ctx.main_graph();
        graph.get_node(p.node_id).unwrap().name.as_ref().unwrap().clone()
    }).collect();

    println!("\n--- НАЧАЛО ЦИКЛА ОБУЧЕНИЯ ---\n");
    let start_time = Instant::now();

    for epoch in 0..15 {
        let device_data = backend.load_data(&runtime_data).unwrap();
        
        let mut initial_memo: Memo<B::DeviceData> = HashMap::new();
        for (name, data) in device_data {
            if let Some(node_id) = forward_graph.nodes.values()
                .find(|n| match &n.node_type {
                    NodeType::Input{name: n_name} | NodeType::Parameter{name: n_name} => n_name == &name,
                    _ => false
                }).map(|n| n.id) {
                initial_memo.insert((forward_graph.id, node_id), data);
            }
        }

        let (loss_device_vec, forward_memo) = backend.run(&forward_graph, initial_memo).unwrap();
        let loss_value_vec = backend.retrieve_data(&loss_device_vec).unwrap();
        let loss_value = loss_value_vec.first().unwrap();

        let (grad_device_vec, _grad_memo) = backend.run(&grad_graph, forward_memo).unwrap();
        let grad_value_vec = backend.retrieve_data(&grad_device_vec).unwrap();
        
        let mut computed_grads = HashMap::new();
        for (name, value) in param_names.iter().zip(grad_value_vec.into_iter()) {
            // --- ИСПРАВЛЕНИЕ: Удаляем временный хак ---
            // Старый код: if name.contains("gamma") || name.contains("beta") { continue; }
            computed_grads.insert(name.clone(), value);
        }

        optimizer.step(&mut runtime_data, &computed_grads);

        if let Value::Tensor(loss_tensor) = loss_value {
            println!("Эпоха: {:<2}, Потери (Loss): {:.6}", epoch + 1, loss_tensor.first().unwrap_or(&-1.0));
        }
    }

    let duration = start_time.elapsed();
    println!("\n--- ОБУЧЕНИЕ ЗАВЕРШЕНО ЗА {:.2?} ---", duration);
}