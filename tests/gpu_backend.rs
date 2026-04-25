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
            .find(|(_, node)| matches!(&node.node_type, NodeType::Input { name: n } if n == &name))
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
            .find(|(_, node)| matches!(&node.node_type, NodeType::Input { name: n } if n == &name))
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
            .find(|(_, node)| matches!(&node.node_type, NodeType::Input { name: n } if n == &name))
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
            .find(|(_, node)| matches!(&node.node_type, NodeType::Input { name: n } if n == &name))
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
    Tensor {
        node_id,
        context: std::rc::Rc::clone(ctx),
    }
}

#[test]
fn gpu_conv2d_basic() {
    // Input: [1, 1, 4, 4], Weight: [1, 1, 3, 3]
    let input_data =
        Array4::<f32>::from_shape_fn((1, 1, 4, 4), |(_, _, h, w)| (h * 4 + w) as f32).into_dyn();

    let weight_data =
        Array4::<f32>::from_shape_fn((1, 1, 3, 3), |(_, _, h, w)| (h * 3 + w + 1) as f32 / 10.0)
            .into_dyn();

    // Build graph
    let context = Rc::new(RefCell::new(GraphContext::new()));
    let input_tensor = Tensor::new_input(&context, "input");
    let weight_tensor = Tensor::new_input(&context, "weight");
    let output = create_conv2d_op(&context, &input_tensor, &weight_tensor, (1, 1), (0, 0));
    context
        .borrow_mut()
        .main_graph_mut()
        .set_output(output.node_id);

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
            .find(|(_, node)| matches!(&node.node_type, NodeType::Input { name: n } if n == &name))
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
            .find(|(_, node)| matches!(&node.node_type, NodeType::Input { name: n } if n == &name))
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
    let input_data =
        Array4::<f32>::from_shape_fn((1, 1, 4, 4), |(_, _, h, w)| (h * 4 + w + 1) as f32 / 16.0)
            .into_dyn();

    let weight_data = Array4::<f32>::from_shape_fn((2, 1, 3, 3), |(oc, _, h, w)| {
        ((oc * 9 + h * 3 + w + 1) as f32) / 20.0
    })
    .into_dyn();

    let context = Rc::new(RefCell::new(GraphContext::new()));
    let input_tensor = Tensor::new_input(&context, "input");
    let weight_tensor = Tensor::new_input(&context, "weight");
    let output = create_conv2d_op(&context, &input_tensor, &weight_tensor, (1, 1), (1, 1));
    context
        .borrow_mut()
        .main_graph_mut()
        .set_output(output.node_id);

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
        let node_id = graph
            .nodes
            .iter()
            .find(|(_, node)| matches!(&node.node_type, NodeType::Input { name: n } if n == &name))
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
            .find(|(_, node)| matches!(&node.node_type, NodeType::Input { name: n } if n == &name))
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

    assert_tensors_close(&cpu_tensor, &gpu_tensor, "Conv2d with padding");
    println!("Conv2d with padding: PASSED");
}

#[test]
fn gpu_conv2d_grouped() {
    // Depthwise convolution: groups == in_channels == out_channels.
    // Input [1, 4, 4, 4], weight [4, 1, 3, 3], pad=1 -> output [1, 4, 4, 4].
    let input_data = Array4::<f32>::from_shape_fn((1, 4, 4, 4), |(_, c, h, w)| {
        ((c * 16 + h * 4 + w + 1) as f32) / 16.0
    })
    .into_dyn();
    let weight_data = Array4::<f32>::from_shape_fn((4, 1, 3, 3), |(oc, _, h, w)| {
        ((oc * 9 + h * 3 + w + 1) as f32) / 12.0
    })
    .into_dyn();

    let context = Rc::new(RefCell::new(GraphContext::new()));
    let input_tensor = Tensor::new_input(&context, "input");
    let weight_tensor = Tensor::new_input(&context, "weight");

    let conv_id = context.borrow_mut().main_graph_mut().add_node(
        None,
        NodeType::Conv2d {
            input: input_tensor.node_id,
            weight: weight_tensor.node_id,
            bias: None,
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
            groups: 4,
        },
    );
    context.borrow_mut().main_graph_mut().set_output(conv_id);

    let mut shapes = HashMap::new();
    shapes.insert("input".to_string(), (vec![1, 4, 4, 4], DType::F32));
    shapes.insert("weight".to_string(), (vec![4, 1, 3, 3], DType::F32));
    let mut graph = context.borrow().main_graph().clone();
    ShapeInference::run(&mut graph, &shapes).expect("Shape inference failed");

    let mut data = HashMap::new();
    data.insert("input".to_string(), Value::Tensor(input_data));
    data.insert("weight".to_string(), Value::Tensor(weight_data));

    let cpu = CpuBackend::new();
    let cpu_device = cpu.load_data(&data).unwrap();
    let mut cpu_memo = HashMap::new();
    for (name, value) in cpu_device {
        let nid = graph
            .nodes
            .iter()
            .find(|(_, n)| matches!(&n.node_type, NodeType::Input { name: nn } if nn == &name))
            .map(|(id, _)| *id)
            .unwrap();
        cpu_memo.insert((graph.id, nid), value);
    }
    let (cpu_res, _) = cpu.run(&graph, cpu_memo).unwrap();
    let cpu_t = match &cpu.retrieve_data(&cpu_res).unwrap()[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!(),
    };

    let gpu = pollster::block_on(WgpuBackend::new());
    let gpu_device = gpu.load_data(&data).unwrap();
    let mut gpu_memo = HashMap::new();
    for (name, value) in gpu_device {
        let nid = graph
            .nodes
            .iter()
            .find(|(_, n)| matches!(&n.node_type, NodeType::Input { name: nn } if nn == &name))
            .map(|(id, _)| *id)
            .unwrap();
        gpu_memo.insert((graph.id, nid), value);
    }
    let (gpu_res, _) = gpu.run(&graph, gpu_memo).unwrap();
    let gpu_t = match &gpu.retrieve_data(&gpu_res).unwrap()[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!(),
    };

    assert_tensors_close(&cpu_t, &gpu_t, "Conv2d depthwise (groups=in_channels)");
    println!("Conv2d depthwise: PASSED");
}

#[test]
fn gpu_conv2d_dilated() {
    // Dilated convolution: 5x5 input, 3x3 kernel, dilation=2 -> output 1x1.
    let input_data =
        Array4::<f32>::from_shape_fn((1, 1, 5, 5), |(_, _, h, w)| (h * 5 + w) as f32).into_dyn();
    let weight_data =
        Array4::<f32>::from_shape_fn((1, 1, 3, 3), |(_, _, h, w)| (h * 3 + w + 1) as f32)
            .into_dyn();

    let context = Rc::new(RefCell::new(GraphContext::new()));
    let input_tensor = Tensor::new_input(&context, "input");
    let weight_tensor = Tensor::new_input(&context, "weight");

    let conv_id = context.borrow_mut().main_graph_mut().add_node(
        None,
        NodeType::Conv2d {
            input: input_tensor.node_id,
            weight: weight_tensor.node_id,
            bias: None,
            stride: (1, 1),
            padding: (0, 0),
            dilation: (2, 2),
            groups: 1,
        },
    );
    context.borrow_mut().main_graph_mut().set_output(conv_id);

    let mut shapes = HashMap::new();
    shapes.insert("input".to_string(), (vec![1, 1, 5, 5], DType::F32));
    shapes.insert("weight".to_string(), (vec![1, 1, 3, 3], DType::F32));
    let mut graph = context.borrow().main_graph().clone();
    ShapeInference::run(&mut graph, &shapes).expect("Shape inference failed");

    let mut data = HashMap::new();
    data.insert("input".to_string(), Value::Tensor(input_data));
    data.insert("weight".to_string(), Value::Tensor(weight_data));

    let cpu = CpuBackend::new();
    let cpu_device = cpu.load_data(&data).unwrap();
    let mut cpu_memo = HashMap::new();
    for (name, value) in cpu_device {
        let nid = graph
            .nodes
            .iter()
            .find(|(_, n)| matches!(&n.node_type, NodeType::Input { name: nn } if nn == &name))
            .map(|(id, _)| *id)
            .unwrap();
        cpu_memo.insert((graph.id, nid), value);
    }
    let (cpu_res, _) = cpu.run(&graph, cpu_memo).unwrap();
    let cpu_t = match &cpu.retrieve_data(&cpu_res).unwrap()[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!(),
    };

    let gpu = pollster::block_on(WgpuBackend::new());
    let gpu_device = gpu.load_data(&data).unwrap();
    let mut gpu_memo = HashMap::new();
    for (name, value) in gpu_device {
        let nid = graph
            .nodes
            .iter()
            .find(|(_, n)| matches!(&n.node_type, NodeType::Input { name: nn } if nn == &name))
            .map(|(id, _)| *id)
            .unwrap();
        gpu_memo.insert((graph.id, nid), value);
    }
    let (gpu_res, _) = gpu.run(&graph, gpu_memo).unwrap();
    let gpu_t = match &gpu.retrieve_data(&gpu_res).unwrap()[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!(),
    };

    assert_tensors_close(&cpu_t, &gpu_t, "Conv2d dilated (dilation=2)");
    println!("Conv2d dilated: PASSED");
}

// ============================================================
// LayerNorm GPU tests
// ============================================================

/// Run a LayerNorm-related graph on both backends and compare.
fn run_ln_graph(
    shape: Vec<usize>,
    gamma_shape: Vec<usize>,
    beta_shape: Vec<usize>,
    build: impl Fn(&Rc<RefCell<GraphContext>>, &Tensor, &Tensor, &Tensor) -> Tensor,
    data: HashMap<String, Value>,
    op_name: &str,
) {
    let context = Rc::new(RefCell::new(GraphContext::new()));
    let x = Tensor::new_input(&context, "x");
    let gamma = Tensor::new_input(&context, "gamma");
    let beta = Tensor::new_input(&context, "beta");
    let out = build(&context, &x, &gamma, &beta);
    context
        .borrow_mut()
        .main_graph_mut()
        .set_output(out.node_id);

    let mut shapes = HashMap::new();
    shapes.insert("x".to_string(), (shape, DType::F32));
    shapes.insert("gamma".to_string(), (gamma_shape, DType::F32));
    shapes.insert("beta".to_string(), (beta_shape, DType::F32));
    let mut graph = context.borrow().main_graph().clone();
    ShapeInference::run(&mut graph, &shapes).expect("Shape inference failed");

    let cpu_backend = CpuBackend::new();
    let cpu_device = cpu_backend.load_data(&data).unwrap();
    let mut cpu_memo = HashMap::new();
    for (name, value) in cpu_device {
        let node_id = graph
            .nodes
            .iter()
            .find(|(_, node)| matches!(&node.node_type, NodeType::Input { name: n } if n == &name))
            .map(|(id, _)| *id)
            .unwrap();
        cpu_memo.insert((graph.id, node_id), value);
    }
    let (cpu_results, _) = cpu_backend.run(&graph, cpu_memo).unwrap();
    let cpu_tensor = match &cpu_backend.retrieve_data(&cpu_results).unwrap()[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!("expected tensor"),
    };

    let gpu_backend = pollster::block_on(WgpuBackend::new());
    let gpu_device = gpu_backend.load_data(&data).unwrap();
    let mut gpu_memo = HashMap::new();
    for (name, value) in gpu_device {
        let node_id = graph
            .nodes
            .iter()
            .find(|(_, node)| matches!(&node.node_type, NodeType::Input { name: n } if n == &name))
            .map(|(id, _)| *id)
            .unwrap();
        gpu_memo.insert((graph.id, node_id), value);
    }
    let (gpu_results, _) = gpu_backend.run(&graph, gpu_memo).unwrap();
    let gpu_tensor = match &gpu_backend.retrieve_data(&gpu_results).unwrap()[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!("expected tensor"),
    };

    assert_tensors_close(&cpu_tensor, &gpu_tensor, op_name);
    println!("{}: PASSED", op_name);
}

#[test]
fn gpu_layer_norm_forward() {
    let x = array![[1.0_f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]].into_dyn();
    let gamma = array![[0.5_f32, 1.0, 1.5, 2.0]].into_dyn();
    let beta = array![[0.1_f32, -0.2, 0.3, -0.4]].into_dyn();

    let mut data = HashMap::new();
    data.insert("x".to_string(), Value::Tensor(x));
    data.insert("gamma".to_string(), Value::Tensor(gamma));
    data.insert("beta".to_string(), Value::Tensor(beta));

    run_ln_graph(
        vec![2, 4],
        vec![1, 4],
        vec![1, 4],
        |ctx, x, gamma, beta| {
            let node_id = ctx.borrow_mut().main_graph_mut().add_node(
                None,
                NodeType::LayerNorm {
                    input: x.node_id,
                    gamma: gamma.node_id,
                    beta: beta.node_id,
                    eps: 1e-5,
                },
            );
            Tensor {
                node_id,
                context: Rc::clone(ctx),
            }
        },
        data,
        "LayerNorm forward",
    );
}

#[test]
fn gpu_layer_norm_backward_x() {
    let dy = array![[0.1_f32, -0.2, 0.3, -0.1], [0.2, 0.4, -0.3, 0.05]].into_dyn();
    let x = array![[1.0_f32, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]].into_dyn();
    let gamma = array![[0.5_f32, 1.0, 1.5, 2.0]].into_dyn();

    let mut data = HashMap::new();
    data.insert("x".to_string(), Value::Tensor(x));
    data.insert("gamma".to_string(), Value::Tensor(gamma));
    data.insert(
        "beta".to_string(),
        Value::Tensor(array![[0.0_f32, 0.0, 0.0, 0.0]].into_dyn()),
    );
    data.insert("dy".to_string(), Value::Tensor(dy));

    // We need to build a graph with grad_output + input + gamma + LayerNormBackward.
    let context = Rc::new(RefCell::new(GraphContext::new()));
    let dy_t = Tensor::new_input(&context, "dy");
    let x_t = Tensor::new_input(&context, "x");
    let gamma_t = Tensor::new_input(&context, "gamma");

    let node_id = context.borrow_mut().main_graph_mut().add_node(
        None,
        NodeType::LayerNormBackward {
            grad_output: dy_t.node_id,
            input: x_t.node_id,
            gamma: gamma_t.node_id,
            eps: 1e-5,
        },
    );
    context.borrow_mut().main_graph_mut().set_output(node_id);

    let mut shapes = HashMap::new();
    shapes.insert("dy".to_string(), (vec![2, 4], DType::F32));
    shapes.insert("x".to_string(), (vec![2, 4], DType::F32));
    shapes.insert("gamma".to_string(), (vec![1, 4], DType::F32));
    let mut graph = context.borrow().main_graph().clone();
    ShapeInference::run(&mut graph, &shapes).expect("shape inference");

    data.remove("beta"); // unused here

    let cpu = CpuBackend::new();
    let cpu_device = cpu.load_data(&data).unwrap();
    let mut cpu_memo = HashMap::new();
    for (name, v) in cpu_device {
        let nid = graph
            .nodes
            .iter()
            .find(|(_, n)| matches!(&n.node_type, NodeType::Input { name: nn } if nn == &name))
            .map(|(id, _)| *id)
            .unwrap();
        cpu_memo.insert((graph.id, nid), v);
    }
    let (cpu_res, _) = cpu.run(&graph, cpu_memo).unwrap();
    let cpu_t = match &cpu.retrieve_data(&cpu_res).unwrap()[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!(),
    };

    let gpu = pollster::block_on(WgpuBackend::new());
    let gpu_device = gpu.load_data(&data).unwrap();
    let mut gpu_memo = HashMap::new();
    for (name, v) in gpu_device {
        let nid = graph
            .nodes
            .iter()
            .find(|(_, n)| matches!(&n.node_type, NodeType::Input { name: nn } if nn == &name))
            .map(|(id, _)| *id)
            .unwrap();
        gpu_memo.insert((graph.id, nid), v);
    }
    let (gpu_res, _) = gpu.run(&graph, gpu_memo).unwrap();
    let gpu_t = match &gpu.retrieve_data(&gpu_res).unwrap()[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!(),
    };

    assert_tensors_close(&cpu_t, &gpu_t, "LayerNormBackward (dx)");
    println!("LayerNormBackward (dx): PASSED");
}

#[test]
fn gpu_layer_norm_grad_gamma() {
    let dy = array![[0.1_f32, -0.2, 0.3, -0.1], [0.2, 0.4, -0.3, 0.05]].into_dyn();
    let x = array![[1.0_f32, 2.0, 3.0, 4.0], [-1.0, 0.0, 1.0, 2.0]].into_dyn();

    let mut data = HashMap::new();
    data.insert("x".to_string(), Value::Tensor(x));
    data.insert("dy".to_string(), Value::Tensor(dy));

    let context = Rc::new(RefCell::new(GraphContext::new()));
    let dy_t = Tensor::new_input(&context, "dy");
    let x_t = Tensor::new_input(&context, "x");
    let node_id = context.borrow_mut().main_graph_mut().add_node(
        None,
        NodeType::LayerNormGradGamma {
            grad_output: dy_t.node_id,
            input: x_t.node_id,
            eps: 1e-5,
        },
    );
    context.borrow_mut().main_graph_mut().set_output(node_id);

    let mut shapes = HashMap::new();
    shapes.insert("dy".to_string(), (vec![2, 4], DType::F32));
    shapes.insert("x".to_string(), (vec![2, 4], DType::F32));
    let mut graph = context.borrow().main_graph().clone();
    ShapeInference::run(&mut graph, &shapes).expect("shape inference");

    let cpu = CpuBackend::new();
    let cpu_device = cpu.load_data(&data).unwrap();
    let mut cpu_memo = HashMap::new();
    for (name, v) in cpu_device {
        let nid = graph
            .nodes
            .iter()
            .find(|(_, n)| matches!(&n.node_type, NodeType::Input { name: nn } if nn == &name))
            .map(|(id, _)| *id)
            .unwrap();
        cpu_memo.insert((graph.id, nid), v);
    }
    let (cpu_res, _) = cpu.run(&graph, cpu_memo).unwrap();
    let cpu_t = match &cpu.retrieve_data(&cpu_res).unwrap()[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!(),
    };

    let gpu = pollster::block_on(WgpuBackend::new());
    let gpu_device = gpu.load_data(&data).unwrap();
    let mut gpu_memo = HashMap::new();
    for (name, v) in gpu_device {
        let nid = graph
            .nodes
            .iter()
            .find(|(_, n)| matches!(&n.node_type, NodeType::Input { name: nn } if nn == &name))
            .map(|(id, _)| *id)
            .unwrap();
        gpu_memo.insert((graph.id, nid), v);
    }
    let (gpu_res, _) = gpu.run(&graph, gpu_memo).unwrap();
    let gpu_t = match &gpu.retrieve_data(&gpu_res).unwrap()[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!(),
    };

    assert_tensors_close(&cpu_t, &gpu_t, "LayerNormGradGamma");
    println!("LayerNormGradGamma: PASSED");
}

// ============================================================
// Slice / Concat / SliceBackward GPU tests
// ============================================================

#[test]
fn gpu_slice_last_axis() {
    // [2, 4] -> slice axis=1, start=1, end=3 -> [2, 2]
    let x = array![[1.0_f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]].into_dyn();
    let mut data = HashMap::new();
    data.insert("x".into(), Value::Tensor(x));

    run_graph_and_compare(
        |ctx| {
            let input_id = ctx.borrow().main_graph().inputs[0];
            ctx.borrow_mut().main_graph_mut().add_node(
                None,
                NodeType::Slice {
                    input: input_id,
                    axis: 1,
                    start: 1,
                    end: 3,
                },
            )
        },
        vec![("x".into(), vec![2, 4])],
        data,
        "Slice (axis=1)",
    );
}

#[test]
fn gpu_slice_first_axis() {
    let x = array![[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]].into_dyn();
    let mut data = HashMap::new();
    data.insert("x".into(), Value::Tensor(x));

    run_graph_and_compare(
        |ctx| {
            let input_id = ctx.borrow().main_graph().inputs[0];
            ctx.borrow_mut().main_graph_mut().add_node(
                None,
                NodeType::Slice {
                    input: input_id,
                    axis: 0,
                    start: 1,
                    end: 3,
                },
            )
        },
        vec![("x".into(), vec![4, 2])],
        data,
        "Slice (axis=0)",
    );
}

#[test]
fn gpu_concat_two_inputs() {
    let a = array![[1.0_f32, 2.0], [3.0, 4.0]].into_dyn();
    let b = array![[5.0_f32, 6.0], [7.0, 8.0]].into_dyn();
    let mut data = HashMap::new();
    data.insert("a".into(), Value::Tensor(a));
    data.insert("b".into(), Value::Tensor(b));

    run_graph_and_compare(
        |ctx| {
            let a_id = ctx.borrow().main_graph().inputs[0];
            let b_id = ctx.borrow().main_graph().inputs[1];
            ctx.borrow_mut().main_graph_mut().add_node(
                None,
                NodeType::Concat {
                    inputs: vec![a_id, b_id],
                    axis: 1,
                },
            )
        },
        vec![("a".into(), vec![2, 2]), ("b".into(), vec![2, 2])],
        data,
        "Concat (axis=1)",
    );
}

#[test]
fn gpu_slice_backward_zero_pads() {
    // grad: [2, 2], axis=1, start=1, full_size=4 -> [2, 4] with [_, g0, g1, _]
    let grad = array![[0.1_f32, 0.2], [0.3, 0.4]].into_dyn();
    let mut data = HashMap::new();
    data.insert("grad".into(), Value::Tensor(grad));

    run_graph_and_compare(
        |ctx| {
            let grad_id = ctx.borrow().main_graph().inputs[0];
            ctx.borrow_mut().main_graph_mut().add_node(
                None,
                NodeType::SliceBackward {
                    grad_output: grad_id,
                    axis: 1,
                    start: 1,
                    full_size: 4,
                },
            )
        },
        vec![("grad".into(), vec![2, 2])],
        data,
        "SliceBackward",
    );
}

// ============================================================
// Embedding GPU tests
// ============================================================

#[test]
fn gpu_embedding_forward() {
    // weight: [5, 3], indices: [4] -> output: [4, 3]
    let weight = ndarray::arr2(&[
        [1.0_f32, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0],
    ])
    .into_dyn();
    let indices = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0_f32, 2.0, 4.0, 1.0]).unwrap();

    let mut data = HashMap::new();
    data.insert("indices".into(), Value::Tensor(indices));
    data.insert("w".into(), Value::Tensor(weight));

    run_graph_and_compare(
        |ctx| {
            let idx_id = ctx.borrow().main_graph().inputs[0];
            let w_id = ctx.borrow().main_graph().inputs[1];
            ctx.borrow_mut().main_graph_mut().add_node(
                None,
                NodeType::Embedding {
                    indices: idx_id,
                    weight: w_id,
                },
            )
        },
        vec![("indices".into(), vec![4]), ("w".into(), vec![5, 3])],
        data,
        "Embedding forward",
    );
}

#[test]
fn gpu_embedding_grad() {
    // grad_out: [4, 3], indices: [4], num_embeddings=5 -> grad_weight: [5, 3]
    let grad = ArrayD::from_shape_vec(
        IxDyn(&[4, 3]),
        vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    )
    .unwrap();
    let indices = ArrayD::from_shape_vec(IxDyn(&[4]), vec![0.0_f32, 2.0, 0.0, 1.0]).unwrap();

    let mut data = HashMap::new();
    data.insert("grad".into(), Value::Tensor(grad));
    data.insert("indices".into(), Value::Tensor(indices));

    run_graph_and_compare(
        |ctx| {
            let grad_id = ctx.borrow().main_graph().inputs[0];
            let idx_id = ctx.borrow().main_graph().inputs[1];
            ctx.borrow_mut().main_graph_mut().add_node(
                None,
                NodeType::EmbeddingGrad {
                    grad_output: grad_id,
                    indices: idx_id,
                    num_embeddings: 5,
                },
            )
        },
        vec![("grad".into(), vec![4, 3]), ("indices".into(), vec![4])],
        data,
        "EmbeddingGrad",
    );
}

// ============================================================
// ConvTranspose2d GPU tests
// ============================================================

#[test]
fn gpu_conv_transpose2d_forward() {
    // input: [1, 1, 2, 2], weight: [1, 1, 3, 3] -> output (no stride): [1, 1, 4, 4]
    let x = Array4::<f32>::from_shape_fn((1, 1, 2, 2), |(_, _, h, w)| (h * 2 + w + 1) as f32)
        .into_dyn();
    let weight =
        Array4::<f32>::from_shape_fn((1, 1, 3, 3), |(_, _, h, w)| ((h * 3 + w + 1) as f32) / 10.0)
            .into_dyn();

    let mut data = HashMap::new();
    data.insert("x".into(), Value::Tensor(x));
    data.insert("w".into(), Value::Tensor(weight));

    run_graph_and_compare(
        |ctx| {
            let x_id = ctx.borrow().main_graph().inputs[0];
            let w_id = ctx.borrow().main_graph().inputs[1];
            ctx.borrow_mut().main_graph_mut().add_node(
                None,
                NodeType::ConvTranspose2d {
                    input: x_id,
                    weight: w_id,
                    bias: None,
                    stride: (1, 1),
                    padding: (0, 0),
                    output_padding: (0, 0),
                    dilation: (1, 1),
                    groups: 1,
                },
            )
        },
        vec![
            ("x".into(), vec![1, 1, 2, 2]),
            ("w".into(), vec![1, 1, 3, 3]),
        ],
        data,
        "ConvTranspose2d forward",
    );
}

#[test]
fn gpu_conv_transpose2d_stride2() {
    // input: [1, 2, 2, 2], weight: [2, 1, 2, 2], stride=(2,2) -> output: [1, 1, 4, 4]
    let x = Array4::<f32>::from_shape_fn((1, 2, 2, 2), |(_, c, h, w)| {
        (c as f32 + 1.0) * ((h * 2 + w + 1) as f32) / 4.0
    })
    .into_dyn();
    let weight = Array4::<f32>::from_shape_fn((2, 1, 2, 2), |(ic, _, h, w)| {
        ((ic * 4 + h * 2 + w + 1) as f32) / 8.0
    })
    .into_dyn();

    let mut data = HashMap::new();
    data.insert("x".into(), Value::Tensor(x));
    data.insert("w".into(), Value::Tensor(weight));

    run_graph_and_compare(
        |ctx| {
            let x_id = ctx.borrow().main_graph().inputs[0];
            let w_id = ctx.borrow().main_graph().inputs[1];
            ctx.borrow_mut().main_graph_mut().add_node(
                None,
                NodeType::ConvTranspose2d {
                    input: x_id,
                    weight: w_id,
                    bias: None,
                    stride: (2, 2),
                    padding: (0, 0),
                    output_padding: (0, 0),
                    dilation: (1, 1),
                    groups: 1,
                },
            )
        },
        vec![
            ("x".into(), vec![1, 2, 2, 2]),
            ("w".into(), vec![2, 1, 2, 2]),
        ],
        data,
        "ConvTranspose2d stride=2",
    );
}

// ============================================================
// Pooling GPU tests
// ============================================================

#[test]
fn gpu_max_pool2d_forward() {
    let x =
        Array4::<f32>::from_shape_fn((1, 1, 4, 4), |(_, _, h, w)| (h * 4 + w) as f32 * 0.3 - 2.0)
            .into_dyn();

    let mut data = HashMap::new();
    data.insert("x".into(), Value::Tensor(x));

    run_graph_and_compare(
        |ctx| {
            let input_id = ctx.borrow().main_graph().inputs[0];
            ctx.borrow_mut().main_graph_mut().add_node(
                None,
                NodeType::MaxPool2d {
                    input: input_id,
                    kernel_size: (2, 2),
                    stride: (2, 2),
                },
            )
        },
        vec![("x".into(), vec![1, 1, 4, 4])],
        data,
        "MaxPool2d forward",
    );
}

#[test]
fn gpu_max_unpool2d_backward() {
    let grad = Array4::<f32>::from_shape_fn((1, 1, 2, 2), |(_, _, h, w)| {
        (h as f32 + w as f32) * 0.5 + 1.0
    })
    .into_dyn();
    let orig =
        Array4::<f32>::from_shape_fn((1, 1, 4, 4), |(_, _, h, w)| (h * 4 + w) as f32 * 0.3 - 2.0)
            .into_dyn();

    let mut data = HashMap::new();
    data.insert("grad".into(), Value::Tensor(grad));
    data.insert("orig".into(), Value::Tensor(orig));

    run_graph_and_compare(
        |ctx| {
            let grad_id = ctx.borrow().main_graph().inputs[0];
            let orig_id = ctx.borrow().main_graph().inputs[1];
            ctx.borrow_mut().main_graph_mut().add_node(
                None,
                NodeType::MaxUnpool2d {
                    input: grad_id,
                    original_input: orig_id,
                    kernel_size: (2, 2),
                    stride: (2, 2),
                },
            )
        },
        vec![
            ("grad".into(), vec![1, 1, 2, 2]),
            ("orig".into(), vec![1, 1, 4, 4]),
        ],
        data,
        "MaxUnpool2d backward",
    );
}

#[test]
fn gpu_avg_pool2d_forward() {
    let x = Array4::<f32>::from_shape_fn((1, 2, 4, 4), |(_, c, h, w)| {
        (c as f32 + 1.0) * ((h * 4 + w) as f32) / 16.0
    })
    .into_dyn();

    let mut data = HashMap::new();
    data.insert("x".into(), Value::Tensor(x));

    run_graph_and_compare(
        |ctx| {
            let input_id = ctx.borrow().main_graph().inputs[0];
            ctx.borrow_mut().main_graph_mut().add_node(
                None,
                NodeType::AvgPool2d {
                    input: input_id,
                    kernel_size: (2, 2),
                    stride: (2, 2),
                    padding: (0, 0),
                },
            )
        },
        vec![("x".into(), vec![1, 2, 4, 4])],
        data,
        "AvgPool2d forward",
    );
}

#[test]
fn gpu_avg_unpool2d_backward() {
    let grad = Array4::<f32>::from_shape_fn((1, 1, 2, 2), |(_, _, h, w)| {
        (h as f32 + w as f32) * 0.3 + 1.0
    })
    .into_dyn();
    let orig = Array4::<f32>::zeros((1, 1, 4, 4)).into_dyn();

    let mut data = HashMap::new();
    data.insert("grad".into(), Value::Tensor(grad));
    data.insert("orig".into(), Value::Tensor(orig));

    run_graph_and_compare(
        |ctx| {
            let grad_id = ctx.borrow().main_graph().inputs[0];
            let orig_id = ctx.borrow().main_graph().inputs[1];
            ctx.borrow_mut().main_graph_mut().add_node(
                None,
                NodeType::AvgUnpool2d {
                    input: grad_id,
                    original_input: orig_id,
                    kernel_size: (2, 2),
                    stride: (2, 2),
                    padding: (0, 0),
                },
            )
        },
        vec![
            ("grad".into(), vec![1, 1, 2, 2]),
            ("orig".into(), vec![1, 1, 4, 4]),
        ],
        data,
        "AvgUnpool2d backward",
    );
}

#[test]
fn gpu_adaptive_avg_pool2d_forward() {
    let x = Array4::<f32>::from_shape_fn((1, 3, 7, 7), |(_, c, h, w)| {
        ((c * 49 + h * 7 + w) as f32) / 100.0
    })
    .into_dyn();

    let mut data = HashMap::new();
    data.insert("x".into(), Value::Tensor(x));

    run_graph_and_compare(
        |ctx| {
            let input_id = ctx.borrow().main_graph().inputs[0];
            ctx.borrow_mut().main_graph_mut().add_node(
                None,
                NodeType::AdaptiveAvgPool2d {
                    input: input_id,
                    output_size: (2, 2),
                },
            )
        },
        vec![("x".into(), vec![1, 3, 7, 7])],
        data,
        "AdaptiveAvgPool2d forward",
    );
}

// ============================================================
// Conv2d backward GPU tests
// ============================================================

/// Runs a graph that ends with a `Conv2dBackward*` node on both backends.
fn run_graph_and_compare(
    build: impl Fn(&Rc<RefCell<GraphContext>>) -> rustyasg::asg::NodeId,
    input_shapes: Vec<(String, Vec<usize>)>,
    data: HashMap<String, Value>,
    op_name: &str,
) {
    let context = Rc::new(RefCell::new(GraphContext::new()));
    for (name, _) in &input_shapes {
        let _ = Tensor::new_input(&context, name);
    }
    let out_id = build(&context);
    context.borrow_mut().main_graph_mut().set_output(out_id);

    let mut shapes = HashMap::new();
    for (name, s) in input_shapes {
        shapes.insert(name, (s, DType::F32));
    }
    let mut graph = context.borrow().main_graph().clone();
    ShapeInference::run(&mut graph, &shapes).expect("shape inference");

    let cpu = CpuBackend::new();
    let cpu_device = cpu.load_data(&data).unwrap();
    let mut cpu_memo = HashMap::new();
    for (name, v) in cpu_device {
        let nid = graph
            .nodes
            .iter()
            .find(|(_, n)| matches!(&n.node_type, NodeType::Input { name: nn } if nn == &name))
            .map(|(id, _)| *id)
            .unwrap();
        cpu_memo.insert((graph.id, nid), v);
    }
    let (cpu_res, _) = cpu.run(&graph, cpu_memo).unwrap();
    let cpu_t = match &cpu.retrieve_data(&cpu_res).unwrap()[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!(),
    };

    let gpu = pollster::block_on(WgpuBackend::new());
    let gpu_device = gpu.load_data(&data).unwrap();
    let mut gpu_memo = HashMap::new();
    for (name, v) in gpu_device {
        let nid = graph
            .nodes
            .iter()
            .find(|(_, n)| matches!(&n.node_type, NodeType::Input { name: nn } if nn == &name))
            .map(|(id, _)| *id)
            .unwrap();
        gpu_memo.insert((graph.id, nid), v);
    }
    let (gpu_res, _) = gpu.run(&graph, gpu_memo).unwrap();
    let gpu_t = match &gpu.retrieve_data(&gpu_res).unwrap()[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!(),
    };

    assert_tensors_close(&cpu_t, &gpu_t, op_name);
    println!("{}: PASSED", op_name);
}

#[test]
fn gpu_conv2d_backward_input() {
    // input: [1, 1, 4, 4], weight: [2, 1, 3, 3], grad_out: [1, 2, 2, 2]
    let grad_out = Array4::<f32>::from_shape_fn((1, 2, 2, 2), |(_, oc, h, w)| {
        (oc as f32 + 1.0) * ((h * 2 + w + 1) as f32) / 8.0
    })
    .into_dyn();
    let weight = Array4::<f32>::from_shape_fn((2, 1, 3, 3), |(oc, _, h, w)| {
        ((oc * 9 + h * 3 + w + 1) as f32) / 20.0
    })
    .into_dyn();

    let mut data = HashMap::new();
    data.insert("grad_out".into(), Value::Tensor(grad_out));
    data.insert("weight".into(), Value::Tensor(weight));

    run_graph_and_compare(
        |ctx| {
            let grad_id = ctx.borrow().main_graph().inputs[0];
            let w_id = ctx.borrow().main_graph().inputs[1];
            ctx.borrow_mut().main_graph_mut().add_node(
                None,
                NodeType::Conv2dBackwardInput {
                    grad_output: grad_id,
                    weight: w_id,
                    input_shape: (1, 1, 4, 4),
                    stride: (1, 1),
                    padding: (0, 0),
                    dilation: (1, 1),
                    groups: 1,
                },
            )
        },
        vec![
            ("grad_out".into(), vec![1, 2, 2, 2]),
            ("weight".into(), vec![2, 1, 3, 3]),
        ],
        data,
        "Conv2dBackwardInput",
    );
}

#[test]
fn gpu_conv2d_backward_weight() {
    let grad_out = Array4::<f32>::from_shape_fn((1, 2, 2, 2), |(_, oc, h, w)| {
        (oc as f32 + 1.0) * ((h * 2 + w + 1) as f32) / 8.0
    })
    .into_dyn();
    let x =
        Array4::<f32>::from_shape_fn((1, 1, 4, 4), |(_, _, h, w)| (h * 4 + w + 1) as f32 / 16.0)
            .into_dyn();

    let mut data = HashMap::new();
    data.insert("grad_out".into(), Value::Tensor(grad_out));
    data.insert("x".into(), Value::Tensor(x));

    run_graph_and_compare(
        |ctx| {
            let grad_id = ctx.borrow().main_graph().inputs[0];
            let x_id = ctx.borrow().main_graph().inputs[1];
            ctx.borrow_mut().main_graph_mut().add_node(
                None,
                NodeType::Conv2dBackwardWeight {
                    grad_output: grad_id,
                    input: x_id,
                    weight_shape: (2, 1, 3, 3),
                    stride: (1, 1),
                    padding: (0, 0),
                    dilation: (1, 1),
                    groups: 1,
                },
            )
        },
        vec![
            ("grad_out".into(), vec![1, 2, 2, 2]),
            ("x".into(), vec![1, 1, 4, 4]),
        ],
        data,
        "Conv2dBackwardWeight",
    );
}

#[test]
fn gpu_conv2d_backward_input_with_padding_stride() {
    // input: [1, 2, 5, 5], weight: [3, 2, 3, 3], stride=2, padding=1 -> out: [1, 3, 3, 3]
    let grad_out = Array4::<f32>::from_shape_fn((1, 3, 3, 3), |(_, oc, h, w)| {
        (oc as f32 + 1.0) * (h as f32 + w as f32) * 0.1
    })
    .into_dyn();
    let weight = Array4::<f32>::from_shape_fn((3, 2, 3, 3), |(oc, ic, h, w)| {
        ((oc * 18 + ic * 9 + h * 3 + w + 1) as f32) / 40.0
    })
    .into_dyn();

    let mut data = HashMap::new();
    data.insert("grad_out".into(), Value::Tensor(grad_out));
    data.insert("weight".into(), Value::Tensor(weight));

    run_graph_and_compare(
        |ctx| {
            let grad_id = ctx.borrow().main_graph().inputs[0];
            let w_id = ctx.borrow().main_graph().inputs[1];
            ctx.borrow_mut().main_graph_mut().add_node(
                None,
                NodeType::Conv2dBackwardInput {
                    grad_output: grad_id,
                    weight: w_id,
                    input_shape: (1, 2, 5, 5),
                    stride: (2, 2),
                    padding: (1, 1),
                    dilation: (1, 1),
                    groups: 1,
                },
            )
        },
        vec![
            ("grad_out".into(), vec![1, 3, 3, 3]),
            ("weight".into(), vec![3, 2, 3, 3]),
        ],
        data,
        "Conv2dBackwardInput (stride=2, pad=1)",
    );
}

#[test]
fn gpu_layer_norm_grad_beta() {
    let dy = array![[0.1_f32, -0.2, 0.3, -0.1], [0.2, 0.4, -0.3, 0.05]].into_dyn();

    let mut data = HashMap::new();
    data.insert("dy".to_string(), Value::Tensor(dy));

    let context = Rc::new(RefCell::new(GraphContext::new()));
    let dy_t = Tensor::new_input(&context, "dy");
    let node_id = context.borrow_mut().main_graph_mut().add_node(
        None,
        NodeType::LayerNormGradBeta {
            grad_output: dy_t.node_id,
        },
    );
    context.borrow_mut().main_graph_mut().set_output(node_id);

    let mut shapes = HashMap::new();
    shapes.insert("dy".to_string(), (vec![2, 4], DType::F32));
    let mut graph = context.borrow().main_graph().clone();
    ShapeInference::run(&mut graph, &shapes).expect("shape inference");

    let cpu = CpuBackend::new();
    let cpu_device = cpu.load_data(&data).unwrap();
    let mut cpu_memo = HashMap::new();
    for (name, v) in cpu_device {
        let nid = graph
            .nodes
            .iter()
            .find(|(_, n)| matches!(&n.node_type, NodeType::Input { name: nn } if nn == &name))
            .map(|(id, _)| *id)
            .unwrap();
        cpu_memo.insert((graph.id, nid), v);
    }
    let (cpu_res, _) = cpu.run(&graph, cpu_memo).unwrap();
    let cpu_t = match &cpu.retrieve_data(&cpu_res).unwrap()[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!(),
    };

    let gpu = pollster::block_on(WgpuBackend::new());
    let gpu_device = gpu.load_data(&data).unwrap();
    let mut gpu_memo = HashMap::new();
    for (name, v) in gpu_device {
        let nid = graph
            .nodes
            .iter()
            .find(|(_, n)| matches!(&n.node_type, NodeType::Input { name: nn } if nn == &name))
            .map(|(id, _)| *id)
            .unwrap();
        gpu_memo.insert((graph.id, nid), v);
    }
    let (gpu_res, _) = gpu.run(&graph, gpu_memo).unwrap();
    let gpu_t = match &gpu.retrieve_data(&gpu_res).unwrap()[0] {
        Value::Tensor(t) => t.clone(),
        _ => panic!(),
    };

    assert_tensors_close(&cpu_t, &gpu_t, "LayerNormGradBeta");
    println!("LayerNormGradBeta: PASSED");
}
