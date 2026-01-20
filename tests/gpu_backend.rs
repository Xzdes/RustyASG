//! GPU Backend Tests
//!
//! Tests for verifying GPU (wgpu) backend operations.
//! These tests compare GPU results with CPU backend results.

use ndarray::{array, Array4, ArrayD, IxDyn};
use rustyasg::analysis::shape_inference::ShapeInference;
use rustyasg::asg::{DType, NodeType, Value};
use rustyasg::runtime::backend::Backend;
use rustyasg::runtime::cpu_backend::CpuBackend;
use rustyasg::runtime::wgpu_backend::WgpuBackend;
use rustyasg::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

const TOL: f32 = 1e-5;

/// Compare two tensors element-wise with tolerance
fn assert_tensors_close(cpu: &ArrayD<f32>, gpu: &ArrayD<f32>, op_name: &str) {
    assert_eq!(
        cpu.shape(),
        gpu.shape(),
        "{}: shape mismatch: CPU {:?} vs GPU {:?}",
        op_name,
        cpu.shape(),
        gpu.shape()
    );

    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let diff = (c - g).abs();
        let max_val = c.abs().max(g.abs()).max(1e-7);
        let rel_err = diff / max_val;
        assert!(
            rel_err < TOL || diff < TOL,
            "{}: mismatch at index {}: CPU={}, GPU={}, rel_err={}",
            op_name,
            i,
            c,
            g,
            rel_err
        );
    }
}

/// Helper to run a simple unary operation on both CPU and GPU
fn test_unary_op<F>(op_name: &str, input_data: ArrayD<f32>, build_op: F)
where
    F: Fn(&Tensor) -> Tensor,
{
    // Build graph
    let context = Rc::new(RefCell::new(GraphContext::new()));
    let x = Tensor::new_input(&context, "x");
    let output = build_op(&x);
    context
        .borrow_mut()
        .main_graph_mut()
        .set_output(output.node_id);

    // Shape inference
    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), (input_data.shape().to_vec(), DType::F32));
    let mut graph = context.borrow().main_graph().clone();
    ShapeInference::run(&mut graph, &shapes).expect("Shape inference failed");

    // Prepare data
    let mut data = HashMap::new();
    data.insert("x".to_string(), Value::Tensor(input_data.clone()));

    // Run on CPU
    let cpu_backend = CpuBackend::new();
    let cpu_device_data = cpu_backend.load_data(&data).unwrap();
    let mut cpu_memo = HashMap::new();
    for (name, value) in cpu_device_data {
        let node_id = graph
            .nodes
            .iter()
            .find(|(_, node)| {
                matches!(&node.node_type, NodeType::Input { name: n } if n == &name)
            })
            .map(|(id, _)| *id)
            .unwrap();
        cpu_memo.insert((graph.id, node_id), value);
    }
    let (cpu_results, _) = cpu_backend.run(&graph, cpu_memo).unwrap();
    let cpu_values = cpu_backend.retrieve_data(&cpu_results).unwrap();
    let cpu_tensor = match &cpu_values[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!("Expected tensor"),
    };

    // Run on GPU
    let gpu_backend = pollster::block_on(WgpuBackend::new());
    let gpu_device_data = gpu_backend.load_data(&data).unwrap();
    let mut gpu_memo = HashMap::new();
    for (name, value) in gpu_device_data {
        let node_id = graph
            .nodes
            .iter()
            .find(|(_, node)| {
                matches!(&node.node_type, NodeType::Input { name: n } if n == &name)
            })
            .map(|(id, _)| *id)
            .unwrap();
        gpu_memo.insert((graph.id, node_id), value);
    }
    let (gpu_results, _) = gpu_backend.run(&graph, gpu_memo).unwrap();
    let gpu_values = gpu_backend.retrieve_data(&gpu_results).unwrap();
    let gpu_tensor = match &gpu_values[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!("Expected tensor"),
    };

    // Compare
    assert_tensors_close(&cpu_tensor, &gpu_tensor, op_name);
    println!("{}: PASSED", op_name);
}

/// Helper to run a simple binary operation on both CPU and GPU
fn test_binary_op<F>(op_name: &str, a_data: ArrayD<f32>, b_data: ArrayD<f32>, build_op: F)
where
    F: Fn(&Tensor, &Tensor) -> Tensor,
{
    let context = Rc::new(RefCell::new(GraphContext::new()));
    let a = Tensor::new_input(&context, "a");
    let b = Tensor::new_input(&context, "b");
    let output = build_op(&a, &b);
    context
        .borrow_mut()
        .main_graph_mut()
        .set_output(output.node_id);

    let mut shapes = HashMap::new();
    shapes.insert("a".to_string(), (a_data.shape().to_vec(), DType::F32));
    shapes.insert("b".to_string(), (b_data.shape().to_vec(), DType::F32));
    let mut graph = context.borrow().main_graph().clone();
    ShapeInference::run(&mut graph, &shapes).expect("Shape inference failed");

    let mut data = HashMap::new();
    data.insert("a".to_string(), Value::Tensor(a_data.clone()));
    data.insert("b".to_string(), Value::Tensor(b_data.clone()));

    // CPU
    let cpu_backend = CpuBackend::new();
    let cpu_device_data = cpu_backend.load_data(&data).unwrap();
    let mut cpu_memo = HashMap::new();
    for (name, value) in cpu_device_data {
        let node_id = graph
            .nodes
            .iter()
            .find(|(_, node)| {
                matches!(&node.node_type, NodeType::Input { name: n } if n == &name)
            })
            .map(|(id, _)| *id)
            .unwrap();
        cpu_memo.insert((graph.id, node_id), value);
    }
    let (cpu_results, _) = cpu_backend.run(&graph, cpu_memo).unwrap();
    let cpu_values = cpu_backend.retrieve_data(&cpu_results).unwrap();
    let cpu_tensor = match &cpu_values[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!("Expected tensor"),
    };

    // GPU
    let gpu_backend = pollster::block_on(WgpuBackend::new());
    let gpu_device_data = gpu_backend.load_data(&data).unwrap();
    let mut gpu_memo = HashMap::new();
    for (name, value) in gpu_device_data {
        let node_id = graph
            .nodes
            .iter()
            .find(|(_, node)| {
                matches!(&node.node_type, NodeType::Input { name: n } if n == &name)
            })
            .map(|(id, _)| *id)
            .unwrap();
        gpu_memo.insert((graph.id, node_id), value);
    }
    let (gpu_results, _) = gpu_backend.run(&graph, gpu_memo).unwrap();
    let gpu_values = gpu_backend.retrieve_data(&gpu_results).unwrap();
    let gpu_tensor = match &gpu_values[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!("Expected tensor"),
    };

    assert_tensors_close(&cpu_tensor, &gpu_tensor, op_name);
    println!("{}: PASSED", op_name);
}

// ============================================================
// Unary Operations Tests
// ============================================================

#[test]
fn gpu_relu() {
    let data = array![[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]].into_dyn();
    test_unary_op("ReLU", data, |x| x.relu());
}

#[test]
fn gpu_sigmoid() {
    let data = array![[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]].into_dyn();
    test_unary_op("Sigmoid", data, |x| x.sigmoid());
}

#[test]
fn gpu_tanh() {
    let data = array![[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]].into_dyn();
    test_unary_op("Tanh", data, |x| x.tanh());
}

#[test]
fn gpu_exp() {
    let data = array![[-1.0, 0.0, 1.0], [0.5, -0.5, 2.0]].into_dyn();
    test_unary_op("Exp", data, |x| x.exp());
}

#[test]
fn gpu_log() {
    let data = array![[0.1, 1.0, 2.0], [0.5, 3.0, 10.0]].into_dyn();
    test_unary_op("Log", data, |x| x.log());
}

#[test]
fn gpu_neg() {
    let data = array![[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]].into_dyn();
    test_unary_op("Neg", data, |x| x.neg());
}

#[test]
fn gpu_abs() {
    let data = array![[-1.0, 0.0, 1.0], [-2.0, 3.0, -4.0]].into_dyn();
    test_unary_op("Abs", data, |x| x.abs());
}

#[test]
fn gpu_sqrt() {
    let data = array![[1.0, 4.0, 9.0], [16.0, 25.0, 36.0]].into_dyn();
    test_unary_op("Sqrt", data, |x| x.sqrt());
}

#[test]
fn gpu_gelu() {
    let data = array![[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]].into_dyn();
    test_unary_op("GELU", data, |x| x.gelu());
}

#[test]
fn gpu_silu() {
    let data = array![[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]].into_dyn();
    test_unary_op("SiLU", data, |x| x.silu());
}

#[test]
fn gpu_leaky_relu() {
    let data = array![[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]].into_dyn();
    test_unary_op("LeakyReLU", data, |x| x.leaky_relu(0.01));
}

// ============================================================
// Binary Operations Tests
// ============================================================

#[test]
fn gpu_add() {
    let a = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
    let b = array![[5.0, 6.0], [7.0, 8.0]].into_dyn();
    test_binary_op("Add", a, b, |x, y| x + y);
}

#[test]
fn gpu_subtract() {
    let a = array![[5.0, 6.0], [7.0, 8.0]].into_dyn();
    let b = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
    test_binary_op("Subtract", a, b, |x, y| x - y);
}

#[test]
fn gpu_multiply() {
    let a = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
    let b = array![[2.0, 3.0], [4.0, 5.0]].into_dyn();
    test_binary_op("Multiply", a, b, |x, y| x * y);
}

#[test]
fn gpu_divide() {
    let a = array![[4.0, 6.0], [8.0, 10.0]].into_dyn();
    let b = array![[2.0, 3.0], [4.0, 5.0]].into_dyn();
    test_binary_op("Divide", a, b, |x, y| x / y);
}

#[test]
fn gpu_add_broadcast_scalar() {
    // Test broadcasting: [2, 3] + scalar -> [2, 3]
    // Note: Full row/column broadcasting not yet implemented on GPU
    let a = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
    let b = ArrayD::from_shape_vec(IxDyn(&[1]), vec![10.0]).unwrap();
    test_binary_op("Add (broadcast scalar)", a, b, |x, y| x + y);
}

// ============================================================
// Matrix Operations Tests
// ============================================================

#[test]
fn gpu_matmul() {
    let a = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].into_dyn(); // [3, 2]
    let b = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(); // [2, 3]
    test_binary_op("MatMul", a, b, |x, y| x.matmul(y));
}

// Note: Batched matmul (3D tensors) is not yet supported on CPU backend
// GPU backend supports it but we can't test without CPU reference
// #[test]
// fn gpu_matmul_batched() { ... }

// ============================================================
// Reduction Operations Tests
// ============================================================

#[test]
fn gpu_sum() {
    let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
    test_unary_op("Sum", data, |x| x.sum());
}

#[test]
fn gpu_mean() {
    let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
    test_unary_op("Mean", data, |x| x.mean());
}

#[test]
fn gpu_variance() {
    let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
    test_unary_op("Variance", data, |x| x.variance());
}

// ============================================================
// Softmax Test
// ============================================================

#[test]
fn gpu_softmax() {
    let data = array![[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]].into_dyn();
    test_unary_op("Softmax", data, |x| x.softmax());
}

// ============================================================
// Transpose Test
// ============================================================

#[test]
fn gpu_transpose() {
    let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn(); // [2, 3]
    test_unary_op("Transpose", data, |x| x.transpose(0, 1)); // -> [3, 2]
}

// ============================================================
// Power Test
// ============================================================

#[test]
fn gpu_power_square() {
    let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
    test_unary_op("Power (square)", data, |x| x.pow_scalar(2.0));
}

// ============================================================
// Clamp Test
// ============================================================

#[test]
fn gpu_clamp() {
    let data = array![[-2.0, -1.0, 0.0], [1.0, 2.0, 3.0]].into_dyn();
    test_unary_op("Clamp", data, |x| x.clamp(-0.5, 1.5));
}

// ============================================================
// Conv2d Test
// ============================================================

/// Helper to create Conv2d via NodeType
fn create_conv2d_op(
    ctx: &std::rc::Rc<std::cell::RefCell<rustyasg::tensor::GraphContext>>,
    input: &Tensor,
    weight: &Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Tensor {
    let node_id = ctx.borrow_mut().main_graph_mut().add_node(
        None,
        NodeType::Conv2d {
            input: input.node_id,
            weight: weight.node_id,
            bias: None,
            stride,
            padding,
            dilation: (1, 1),
            groups: 1,
        },
    );
    Tensor { node_id, context: std::rc::Rc::clone(ctx) }
}

#[test]
fn gpu_conv2d_basic() {
    // Input: [1, 1, 4, 4], Weight: [1, 1, 3, 3]
    let input_data = Array4::<f32>::from_shape_fn((1, 1, 4, 4), |(_, _, h, w)| {
        (h * 4 + w) as f32
    }).into_dyn();

    let weight_data = Array4::<f32>::from_shape_fn((1, 1, 3, 3), |(_, _, h, w)| {
        (h * 3 + w + 1) as f32 / 10.0
    }).into_dyn();

    // Build graph
    let context = Rc::new(RefCell::new(GraphContext::new()));
    let input_tensor = Tensor::new_input(&context, "input");
    let weight_tensor = Tensor::new_input(&context, "weight");
    let output = create_conv2d_op(&context, &input_tensor, &weight_tensor, (1, 1), (0, 0));
    context.borrow_mut().main_graph_mut().set_output(output.node_id);

    // Shape inference
    let mut shapes = HashMap::new();
    shapes.insert("input".to_string(), (vec![1, 1, 4, 4], DType::F32));
    shapes.insert("weight".to_string(), (vec![1, 1, 3, 3], DType::F32));
    let mut graph = context.borrow().main_graph().clone();
    ShapeInference::run(&mut graph, &shapes).expect("Shape inference failed");

    // Prepare data
    let mut data = HashMap::new();
    data.insert("input".to_string(), Value::Tensor(input_data.clone()));
    data.insert("weight".to_string(), Value::Tensor(weight_data.clone()));

    // Run on CPU
    let cpu_backend = CpuBackend::new();
    let cpu_device_data = cpu_backend.load_data(&data).unwrap();
    let mut cpu_memo = HashMap::new();
    for (name, value) in cpu_device_data {
        let node_id = graph
            .nodes
            .iter()
            .find(|(_, node)| {
                matches!(&node.node_type, NodeType::Input { name: n } if n == &name)
            })
            .map(|(id, _)| *id)
            .unwrap();
        cpu_memo.insert((graph.id, node_id), value);
    }
    let (cpu_results, _) = cpu_backend.run(&graph, cpu_memo).unwrap();
    let cpu_values = cpu_backend.retrieve_data(&cpu_results).unwrap();
    let cpu_tensor = match &cpu_values[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!("Expected tensor"),
    };

    // Run on GPU
    let gpu_backend = pollster::block_on(WgpuBackend::new());
    let gpu_device_data = gpu_backend.load_data(&data).unwrap();
    let mut gpu_memo = HashMap::new();
    for (name, value) in gpu_device_data {
        let node_id = graph
            .nodes
            .iter()
            .find(|(_, node)| {
                matches!(&node.node_type, NodeType::Input { name: n } if n == &name)
            })
            .map(|(id, _)| *id)
            .unwrap();
        gpu_memo.insert((graph.id, node_id), value);
    }
    let (gpu_results, _) = gpu_backend.run(&graph, gpu_memo).unwrap();
    let gpu_values = gpu_backend.retrieve_data(&gpu_results).unwrap();
    let gpu_tensor = match &gpu_values[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!("Expected tensor"),
    };

    // Compare
    assert_tensors_close(&cpu_tensor, &gpu_tensor, "Conv2d");
    println!("Conv2d: PASSED");
}

#[test]
fn gpu_conv2d_with_padding() {
    // Input: [1, 1, 4, 4], Weight: [2, 1, 3, 3], Padding: (1, 1)
    let input_data = Array4::<f32>::from_shape_fn((1, 1, 4, 4), |(_, _, h, w)| {
        (h * 4 + w + 1) as f32 / 16.0
    }).into_dyn();

    let weight_data = Array4::<f32>::from_shape_fn((2, 1, 3, 3), |(oc, _, h, w)| {
        ((oc * 9 + h * 3 + w + 1) as f32) / 20.0
    }).into_dyn();

    let context = Rc::new(RefCell::new(GraphContext::new()));
    let input_tensor = Tensor::new_input(&context, "input");
    let weight_tensor = Tensor::new_input(&context, "weight");
    let output = create_conv2d_op(&context, &input_tensor, &weight_tensor, (1, 1), (1, 1));
    context.borrow_mut().main_graph_mut().set_output(output.node_id);

    let mut shapes = HashMap::new();
    shapes.insert("input".to_string(), (vec![1, 1, 4, 4], DType::F32));
    shapes.insert("weight".to_string(), (vec![2, 1, 3, 3], DType::F32));
    let mut graph = context.borrow().main_graph().clone();
    ShapeInference::run(&mut graph, &shapes).expect("Shape inference failed");

    let mut data = HashMap::new();
    data.insert("input".to_string(), Value::Tensor(input_data.clone()));
    data.insert("weight".to_string(), Value::Tensor(weight_data.clone()));

    // CPU
    let cpu_backend = CpuBackend::new();
    let cpu_device_data = cpu_backend.load_data(&data).unwrap();
    let mut cpu_memo = HashMap::new();
    for (name, value) in cpu_device_data {
        let node_id = graph.nodes.iter()
            .find(|(_, node)| matches!(&node.node_type, NodeType::Input { name: n } if n == &name))
            .map(|(id, _)| *id).unwrap();
        cpu_memo.insert((graph.id, node_id), value);
    }
    let (cpu_results, _) = cpu_backend.run(&graph, cpu_memo).unwrap();
    let cpu_values = cpu_backend.retrieve_data(&cpu_results).unwrap();
    let cpu_tensor = match &cpu_values[0] { Value::Tensor(t) => t.clone(), _ => panic!("Expected tensor") };

    // GPU
    let gpu_backend = pollster::block_on(WgpuBackend::new());
    let gpu_device_data = gpu_backend.load_data(&data).unwrap();
    let mut gpu_memo = HashMap::new();
    for (name, value) in gpu_device_data {
        let node_id = graph.nodes.iter()
            .find(|(_, node)| matches!(&node.node_type, NodeType::Input { name: n } if n == &name))
            .map(|(id, _)| *id).unwrap();
        gpu_memo.insert((graph.id, node_id), value);
    }
    let (gpu_results, _) = gpu_backend.run(&graph, gpu_memo).unwrap();
    let gpu_values = gpu_backend.retrieve_data(&gpu_results).unwrap();
    let gpu_tensor = match &gpu_values[0] { Value::Tensor(t) => t.clone(), _ => panic!("Expected tensor") };

    assert_tensors_close(&cpu_tensor, &gpu_tensor, "Conv2d with padding");
    println!("Conv2d with padding: PASSED");
}
