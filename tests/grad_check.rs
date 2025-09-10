// --- Файл: tests/grad_check.rs ---

//! Интеграционный тест для проверки корректности автоматического дифференцирования.

use rustyasg::analysis::shape_inference::ShapeInference;
use rustyasg::asg::{DType, Value};
use rustyasg::autograd::Gradients;
use rustyasg::runtime::backend::{Backend, Memo};
use rustyasg::runtime::cpu_backend::CpuBackend;
use rustyasg::tensor::{GraphContext, Tensor};
use rustyasg::nn::{LayerNorm, Module};

use ndarray::ArrayD;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

const EPSILON: f32 = 1e-4;
const TOLERANCE: f32 = 5e-2; // Допуск чуть выше для сложных операций

/// Сравнивает два тензора поэлементно и паникует, если они не близки.
fn assert_grads_are_close(analytic: &ArrayD<f32>, numeric: &ArrayD<f32>, tolerance: f32, name: &str) {
    assert_eq!(analytic.shape(), numeric.shape(), "Gradient shapes for '{}' do not match!", name);

    for (a, n) in analytic.iter().zip(numeric.iter()) {
        let diff = (a - n).abs();
        let larger = a.abs().max(n.abs());
        if larger == 0.0 { continue; }
        let relative_error = diff / larger;
        if relative_error > tolerance {
            panic!(
                "Gradients for '{}' do not match! Analytic: {:.6}, Numeric: {:.6}, Relative Error: {:.6}",
                name, a, n, relative_error
            );
        }
    }
}

// --- НОВЫЙ МНОГОВАРИАНТНЫЙ ГРАД-ЧЕКЕР ---

/// Структура для проверки градиентов функций с несколькими входами.
struct GradChecker {
    // Функция, строящая граф. Принимает HashMap входов.
    graph_builder: Box<dyn Fn(&HashMap<String, Tensor>) -> Tensor>,
    // Начальные значения для всех входов.
    initial_values: HashMap<String, ArrayD<f32>>,
    // Имя входа, для которого вычисляется градиент.
    wrt_name: String,
}

impl GradChecker {
    /// Вычисляет аналитический градиент для `wrt_name`.
    fn get_analytic_grad(&self) -> ArrayD<f32> {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let backend = CpuBackend::new();

        let mut input_tensors = HashMap::new();
        let mut initial_shapes = HashMap::new();
        for (name, value) in &self.initial_values {
            input_tensors.insert(name.clone(), Tensor::new_input(&context, name));
            initial_shapes.insert(name.clone(), (value.shape().to_vec(), DType::F32));
        }

        let y_tensor = (self.graph_builder)(&input_tensors);
        let mut forward_graph = context.borrow().main_graph().clone();
        forward_graph.set_output(y_tensor.node_id);
        ShapeInference::run(&mut forward_graph, &initial_shapes).unwrap();

        let wrt_tensor = input_tensors.get(&self.wrt_name).unwrap();
        let grad_generator = Gradients::new(forward_graph.clone());
        let mut grad_graph = grad_generator.build(y_tensor.node_id, &[wrt_tensor.node_id]).unwrap();

        let mut grad_initial_shapes = HashMap::new();
        for node in forward_graph.nodes.values() {
            if let Some(shape) = &node.shape {
                let external_node_name = format!("external_{}_{}", forward_graph.id, node.id);
                grad_initial_shapes.insert(external_node_name, (shape.clone(), DType::F32));
            }
        }
        ShapeInference::run(&mut grad_graph, &grad_initial_shapes).unwrap();

        let device_data = backend.load_data(&self.initial_values.iter().map(|(k, v)| (k.clone(), Value::Tensor(v.clone()))).collect()).unwrap();
        let mut initial_memo: Memo<Value> = HashMap::new();
        for (name, tensor) in &input_tensors {
            initial_memo.insert((forward_graph.id, tensor.node_id), device_data.get(name).unwrap().clone());
        }

        let (_, forward_memo) = backend.run(&forward_graph, initial_memo).unwrap();
        let (grad_outputs, _) = backend.run(&grad_graph, forward_memo).unwrap();
        let grad_value = backend.retrieve_data(&grad_outputs).unwrap();

        if let Value::Tensor(grad_tensor) = grad_value.first().unwrap() {
            grad_tensor.clone()
        } else {
            panic!("Gradient is not a tensor");
        }
    }

    /// Вычисляет численный градиент для `wrt_name`.
    fn get_numeric_grad(&self) -> ArrayD<f32> {
        let backend = CpuBackend::new();
        let initial_wrt = self.initial_values.get(&self.wrt_name).unwrap();
        let mut grad = ArrayD::zeros(initial_wrt.shape());

        for i in 0..initial_wrt.len() {
            let mut inputs_plus = self.initial_values.clone();
            inputs_plus.get_mut(&self.wrt_name).unwrap().as_slice_mut().unwrap()[i] += EPSILON;
            let y_plus = self.run_forward_pass(&backend, &inputs_plus);

            let mut inputs_minus = self.initial_values.clone();
            inputs_minus.get_mut(&self.wrt_name).unwrap().as_slice_mut().unwrap()[i] -= EPSILON;
            let y_minus = self.run_forward_pass(&backend, &inputs_minus);

            grad.as_slice_mut().unwrap()[i] = (y_plus - y_minus) / (2.0 * EPSILON);
        }
        grad
    }

    fn run_forward_pass(&self, backend: &CpuBackend, inputs: &HashMap<String, ArrayD<f32>>) -> f32 {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let mut input_tensors = HashMap::new();
        let mut initial_shapes = HashMap::new();
        for (name, value) in inputs {
            input_tensors.insert(name.clone(), Tensor::new_input(&context, name));
            initial_shapes.insert(name.clone(), (value.shape().to_vec(), DType::F32));
        }

        let y_tensor = (self.graph_builder)(&input_tensors);
        let mut graph = context.borrow().main_graph().clone();
        graph.set_output(y_tensor.node_id);
        ShapeInference::run(&mut graph, &initial_shapes).unwrap();

        let runtime_data: HashMap<String, Value> = inputs.iter().map(|(k, v)| (k.clone(), Value::Tensor(v.clone()))).collect();
        let device_data = backend.load_data(&runtime_data).unwrap();

        let mut memo: Memo<Value> = HashMap::new();
        for (name, tensor) in &input_tensors {
            memo.insert((graph.id, tensor.node_id), device_data.get(name).unwrap().clone());
        }

        let (result, _) = backend.run(&graph, memo).unwrap();
        let value = backend.retrieve_data(&result).unwrap();
        if let Value::Tensor(tensor) = value.first().unwrap() {
            *tensor.first().unwrap()
        } else {
            panic!("Result is not a tensor");
        }
    }
}

// --- НОВЫЙ ТЕСТ ДЛЯ LAYER NORM ---
#[test]
fn test_grad_layernorm() {
    // Определяем функцию построения графа
    // Используем `move`, чтобы контекст переместился ВНУТРЬ замыкания.
let layernorm_fn = move |inputs: &HashMap<String, Tensor>| {
    let context = inputs.values().next().unwrap().context.clone();

    let x = inputs.get("x").unwrap();
    
    // Создаем модуль LayerNorm и используем его параметры ИЗ ТЕСТА
    let mut ln_module = LayerNorm::new(&context, "test_ln");
    ln_module.gamma = inputs.get("gamma").unwrap().clone();
    ln_module.beta = inputs.get("beta").unwrap().clone();
    
    // Вызываем НАСТОЯЩИЙ forward pass
    let final_output = ln_module.forward(x);

    // Суммируем для получения скаляра
    final_output.sum()
};

    // Определяем входные данные
    let initial_values = HashMap::from([
        ("x".to_string(), ArrayD::from_shape_vec(ndarray::IxDyn(&[2, 3]), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()),
        ("gamma".to_string(), ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.5, 1.0, 1.5]).unwrap()),
        ("beta".to_string(), ArrayD::from_shape_vec(ndarray::IxDyn(&[3]), vec![0.1, -0.2, 0.3]).unwrap()),
    ]);
    
    // --- 1. Проверяем градиент по входу 'x' ---
    println!("\n--- Тест градиента LayerNorm по 'x' ---");
    let checker_x = GradChecker {
        graph_builder: Box::new(layernorm_fn),
        initial_values: initial_values.clone(),
        wrt_name: "x".to_string(),
    };
    let analytic_grad_x = checker_x.get_analytic_grad();
    let numeric_grad_x = checker_x.get_numeric_grad();
    println!("Аналитический (x): {:?}", analytic_grad_x.as_slice().unwrap());
    println!("Численный    (x): {:?}", numeric_grad_x.as_slice().unwrap());
    assert_grads_are_close(&analytic_grad_x, &numeric_grad_x, TOLERANCE, "x");

    // --- 2. Проверяем градиент по 'gamma' ---
    println!("\n--- Тест градиента LayerNorm по 'gamma' ---");
    let checker_gamma = GradChecker {
        graph_builder: Box::new(layernorm_fn),
        initial_values: initial_values.clone(),
        wrt_name: "gamma".to_string(),
    };
    let analytic_grad_gamma = checker_gamma.get_analytic_grad();
    let numeric_grad_gamma = checker_gamma.get_numeric_grad();
    println!("Аналитический (gamma): {:?}", analytic_grad_gamma.as_slice().unwrap());
    println!("Численный    (gamma): {:?}", numeric_grad_gamma.as_slice().unwrap());
    assert_grads_are_close(&analytic_grad_gamma, &numeric_grad_gamma, TOLERANCE, "gamma");

    // --- 3. Проверяем градиент по 'beta' ---
    println!("\n--- Тест градиента LayerNorm по 'beta' ---");
    let checker_beta = GradChecker {
        graph_builder: Box::new(layernorm_fn),
        initial_values: initial_values.clone(),
        wrt_name: "beta".to_string(),
    };
    let analytic_grad_beta = checker_beta.get_analytic_grad();
    let numeric_grad_beta = checker_beta.get_numeric_grad();
    println!("Аналитический (beta): {:?}", analytic_grad_beta.as_slice().unwrap());
    println!("Численный    (beta): {:?}", numeric_grad_beta.as_slice().unwrap());
    assert_grads_are_close(&analytic_grad_beta, &numeric_grad_beta, TOLERANCE, "beta");
}