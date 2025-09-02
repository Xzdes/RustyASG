//! Интеграционный тест для проверки корректности автоматического дифференцирования.

use rustyasg::analysis::shape_inference::ShapeInference;
use rustyasg::asg::{DType, Value};
use rustyasg::autograd::Gradients;
use rustyasg::runtime::backend::{Backend, Memo};
use rustyasg::runtime::cpu_backend::CpuBackend;
use rustyasg::tensor::{GraphContext, Tensor};

use ndarray::ArrayD;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

const EPSILON: f32 = 1e-4;
const TOLERANCE: f32 = 1e-2;

// --- НАША СОБСТВЕННАЯ ФУНКЦИЯ СРАВНЕНИЯ ---
/// Сравнивает два тензора поэлементно и паникует, если они не близки.
fn assert_grads_are_close(analytic: &ArrayD<f32>, numeric: &ArrayD<f32>, tolerance: f32) {
    assert_eq!(analytic.shape(), numeric.shape(), "Gradient shapes do not match!");

    for (a, n) in analytic.iter().zip(numeric.iter()) {
        // Calculate relative error
        let diff = (a - n).abs();
        let larger = a.abs().max(n.abs());
        
        // Avoid division by zero if both are zero. If they are, they are close enough.
        if larger == 0.0 {
            continue;
        }

        let relative_error = diff / larger;

        if relative_error > tolerance {
            panic!(
                "Gradients do not match! Analytic: {:.6}, Numeric: {:.6}, Relative Error: {:.6}",
                a,
                n,
                relative_error
            );
        }
    }
}

/// Вычисляет аналитический градиент с помощью нашего фреймворка.
fn get_analytic_grad(
    graph_builder: fn(&Tensor) -> Tensor,
    initial_x: &ArrayD<f32>,
) -> ArrayD<f32> {
    // ... (код этой функции остается без изменений)
    let context = Rc::new(RefCell::new(GraphContext::new()));
    let backend = CpuBackend::new();
    let x_tensor = Tensor::new_input(&context, "x");
    let y_tensor = graph_builder(&x_tensor);
    let mut forward_graph = context.borrow().main_graph().clone();
    forward_graph.set_output(y_tensor.node_id);
    let mut initial_shapes = HashMap::new();
    initial_shapes.insert("x".to_string(), (initial_x.shape().to_vec(), DType::F32));
    ShapeInference::run(&mut forward_graph, &initial_shapes).unwrap();
    let grad_generator = Gradients::new(forward_graph.clone());
    let mut grad_graph = grad_generator
        .build(y_tensor.node_id, &[x_tensor.node_id])
        .unwrap();
    let mut grad_initial_shapes = HashMap::new();
     for node in forward_graph.nodes.values() {
        if let Some(shape) = &node.shape {
            let external_node_name = format!("external_{}_{}", forward_graph.id, node.id);
            grad_initial_shapes.insert(external_node_name, (shape.clone(), DType::F32));
        }
    }
    ShapeInference::run(&mut grad_graph, &grad_initial_shapes).unwrap();
    let mut runtime_data = HashMap::new();
    runtime_data.insert("x".to_string(), Value::Tensor(initial_x.clone()));
    let device_data = backend.load_data(&runtime_data).unwrap();
    let mut initial_memo: Memo<Value> = HashMap::new();
    initial_memo.insert((forward_graph.id, x_tensor.node_id), device_data.get("x").unwrap().clone());
    let (_, forward_memo) = backend.run(&forward_graph, initial_memo).unwrap();
    let (grad_outputs, _) = backend.run(&grad_graph, forward_memo).unwrap();
    let grad_value = backend.retrieve_data(&grad_outputs).unwrap();
    if let Value::Tensor(grad_tensor) = grad_value.first().unwrap() {
        grad_tensor.clone()
    } else {
        panic!("Градиент не является тензором");
    }
}

/// Вычисляет численный градиент по методу конечных разностей.
fn get_numeric_grad(
    graph_builder: fn(&Tensor) -> Tensor,
    initial_x: &ArrayD<f32>,
) -> ArrayD<f32> {
    // ... (код этой функции остается без изменений)
    let backend = CpuBackend::new();
    let mut grad = ArrayD::zeros(initial_x.shape());
    for i in 0..initial_x.len() {
        let mut x_plus = initial_x.clone();
        x_plus.as_slice_mut().unwrap()[i] += EPSILON;
        let y_plus = run_forward_pass(&backend, graph_builder, &x_plus);
        let mut x_minus = initial_x.clone();
        x_minus.as_slice_mut().unwrap()[i] -= EPSILON;
        let y_minus = run_forward_pass(&backend, graph_builder, &x_minus);
        let grad_component = (y_plus - y_minus) / (2.0 * EPSILON);
        grad.as_slice_mut().unwrap()[i] = grad_component;
    }
    grad
}

/// Вспомогательная функция для выполнения только прямого прохода.
fn run_forward_pass(
    backend: &CpuBackend,
    graph_builder: fn(&Tensor) -> Tensor,
    input_data: &ArrayD<f32>,
) -> f32 {
    // ... (код этой функции остается без изменений)
    let context = Rc::new(RefCell::new(GraphContext::new()));
    let x_tensor = Tensor::new_input(&context, "x");
    let y_tensor = graph_builder(&x_tensor);
    let mut graph = context.borrow().main_graph().clone();
    graph.set_output(y_tensor.node_id);
    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), (input_data.shape().to_vec(), DType::F32));
    ShapeInference::run(&mut graph, &shapes).unwrap();
    let mut runtime_data = HashMap::new();
    runtime_data.insert("x".to_string(), Value::Tensor(input_data.clone()));
    let device_data = backend.load_data(&runtime_data).unwrap();
    let mut memo: Memo<Value> = HashMap::new();
    memo.insert((graph.id, x_tensor.node_id), device_data.get("x").unwrap().clone());
    let (result, _) = backend.run(&graph, memo).unwrap();
    let value = backend.retrieve_data(&result).unwrap();
    if let Value::Tensor(tensor) = value.first().unwrap() {
        assert_eq!(tensor.len(), 1, "Output for grad check must be a scalar");
        *tensor.first().unwrap()
    } else {
        panic!("Результат не является тензором");
    }
}


#[test]
fn test_grad_multiply() {
    let square_sum_fn = |x: &Tensor| (x * x).sum();
    let x = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
    
    let analytic_grad = get_analytic_grad(square_sum_fn, &x);
    let numeric_grad = get_numeric_grad(square_sum_fn, &x);

    println!("--- Тест для операции Multiply ---");
    println!("Аналитический: {:?}", analytic_grad.as_slice().unwrap());
    println!("Численный:    {:?}", numeric_grad.as_slice().unwrap());
    
    assert_grads_are_close(&analytic_grad, &numeric_grad, TOLERANCE);
}

#[test]
fn test_grad_add_subtract() {
    // Теперь это работает благодаря новым реализациям `impl Sub for Tensor`
    let test_fn = |x: &Tensor| ((x + x) - x).sum();

    let x = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 3]), vec![5.0, -10.0, 1.5]).unwrap();

    let analytic_grad = get_analytic_grad(test_fn, &x);
    let numeric_grad = get_numeric_grad(test_fn, &x);
    
    println!("--- Тест для Add/Subtract ---");
    println!("Аналитический: {:?}", analytic_grad.as_slice().unwrap());
    println!("Численный:    {:?}", numeric_grad.as_slice().unwrap());

    assert_grads_are_close(&analytic_grad, &numeric_grad, TOLERANCE);
}



#[test]
fn test_grad_sum_broadcast() {
    // Тестовая функция: y = (x * C).sum(), где C - постоянный тензор.
    // Градиент должен быть C, транслированный до формы x.
    let test_fn = |x: &Tensor| {
        let const_data = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 3]), vec![10.0, 20.0, 30.0]).unwrap();
        // Здесь мы создаем константу внутри логики построения графа
        let c = Tensor::new_literal(&x.context, const_data, "C");
        (x * c).sum()
    };

    let x = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 3]), vec![1.0, 2.0, 3.0]).unwrap();
    
    let analytic_grad = get_analytic_grad(test_fn, &x);
    let numeric_grad = get_numeric_grad(test_fn, &x);
    
    println!("--- Тест для Sum/Broadcast (градиент Sum) ---");
    println!("Аналитический градиент: {:?}", analytic_grad.as_slice().unwrap());
    println!("Численный градиент:    {:?}", numeric_grad.as_slice().unwrap());
    // Ожидаемый градиент - это сама константа C: [10.0, 20.0, 30.0]

    assert_grads_are_close(&analytic_grad, &numeric_grad, TOLERANCE);
}

// --- ДОБАВЬТЕ ЭТОТ НОВЫЙ ТЕСТ В КОНЕЦ ФАЙЛА ---

#[test]
fn test_grad_complex_ops() {
    // Тестируемая функция: y = (normalized * C).sum()
    // Умножение на константу C гарантирует, что итоговый градиент не будет нулевым.
    let robust_layernorm_fn = |x: &Tensor| {
        let epsilon_const = Tensor::new_literal(&x.context, ndarray::arr0(1e-5).into_dyn(), "epsilon");
        
        let mean = x.mean();
        let x_minus_mean = x - mean;
        
        let variance = x.variance();
        let var_plus_eps = variance + epsilon_const;
        let std_dev = var_plus_eps.sqrt();
        
        let normalized = x_minus_mean / std_dev;

        // --- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ В ТЕСТЕ ---
        // Умножаем на константу, чтобы градиент не был нулевым.
        let constants = Tensor::new_literal(
            &x.context,
            ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 4]), vec![0.1, -0.2, 0.3, -0.4]).unwrap(),
            "C"
        );
        let final_op = normalized * constants;
        
        final_op.sum()
    };

    let x = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 4]), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    
    let analytic_grad = get_analytic_grad(robust_layernorm_fn, &x);
    let numeric_grad = get_numeric_grad(robust_layernorm_fn, &x);
    
    println!("--- Тест для сложных операций (упрощенный LayerNorm) ---");
    println!("Аналитический градиент: {:?}", analytic_grad.as_slice().unwrap());
    println!("Численный градиент:    {:?}", numeric_grad.as_slice().unwrap());
    
    assert_grads_are_close(&analytic_grad, &numeric_grad, TOLERANCE);
}