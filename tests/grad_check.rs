//  tests/grad_check.rs  (финальная компилирующаяся версия)
//! Проверка корректности обратного прохода численным дифференцированием.

use rustyasg::analysis::shape_inference::ShapeInference;
use rustyasg::asg::{DType, NodeType, Value};
use rustyasg::autograd::Gradients;
use rustyasg::nn::{LayerNorm, Module};
use rustyasg::runtime::backend::Backend;
use rustyasg::runtime::cpu_backend::CpuBackend;
use rustyasg::tensor::{GraphContext, Tensor};

use ndarray::{array, Array4, ArrayD};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

const EPS: f32 = 1e-4;          // конечное приращение
const TOL: f32 = 5e-2;          // допустимая относительная погрешность

/// Утилита: сравнивает два тензора и паникует, если не близки.
fn assert_close(a: &ArrayD<f32>, b: &ArrayD<f32>, name: &str) {
    println!(">>> проверка градиентов для '{}'", name);
    println!("    аналитический : {:?}", a.as_slice().unwrap());
    println!("    численный     : {:?}", b.as_slice().unwrap());
    assert_eq!(a.shape(), b.shape(), "shape mismatch для '{}'", name);
    for (i, (av, &nv)) in a.iter().zip(b.iter()).enumerate() {
        let large = av.abs().max(nv.abs());
        if large < 1e-7 { continue; }
        let rel = (av - nv).abs() / large;
        if rel > TOL {
            panic!(
                "\n!!! градиенты различаются для '{}' (idx {})\
                 \n  аналитический: {:.6}\
                 \n  численный    : {:.6}\
                 \n  отн. ошибка  : {:.6} > {}\n",
                name, i, av, nv, rel, TOL
            );
        }
    }
    println!(">>> совпали!\n");
}

/// Структура-тестер: строит граф, считает аналитические и численные градиенты.
struct GradTest {
    builder: Box<dyn Fn(&HashMap<String, Tensor>) -> Tensor>,
    inputs: HashMap<String, ArrayD<f32>>,
    wrt: String,
}

impl GradTest {
    /// Аналитический градиент через autograd.
    fn analytic(&self) -> ArrayD<f32> {
        let ctx = Rc::new(RefCell::new(GraphContext::new()));
        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        for (name, arr) in &self.inputs {
            tensors.insert(name.clone(), Tensor::new_input(&ctx, name));
            shapes.insert(name.clone(), (arr.shape().to_vec(), DType::F32));
        }
        let loss = (self.builder)(&tensors);
        let mut fwd = ctx.borrow().main_graph().clone();
        fwd.set_output(loss.node_id);
        ShapeInference::run(&mut fwd, &shapes).unwrap();

        let wrt_tensor = &tensors[&self.wrt];
        let grad_graph = Gradients::new(fwd.clone())
            .build(loss.node_id, &[wrt_tensor.node_id])
            .unwrap();

        let backend = CpuBackend::new();
        let data: HashMap<String, Value> = self
            .inputs
            .iter()
            .map(|(k, v)| (k.clone(), Value::Tensor(v.clone())))
            .collect();
        let device = backend.load_data(&data).unwrap();
        let mut memo = HashMap::new();
        for (n, t) in &tensors {
            memo.insert((fwd.id, t.node_id), device[n].clone());
        }
        let (_, fwd_memo) = backend.run(&fwd, memo).unwrap();
        let (grad_out, _) = backend.run(&grad_graph, fwd_memo).unwrap();
        let val = backend.retrieve_data(&grad_out).unwrap();
        match val.first().unwrap() {
            Value::Tensor(g) => g.clone(),
            _ => panic!("gradient is not tensor"),
        }
    }

    /// Численный центральный градиент.
    fn numeric(&self) -> ArrayD<f32> {
        let backend = CpuBackend::new();
        let base = &self.inputs[&self.wrt];
        let mut grad = ArrayD::zeros(base.shape());
        for i in 0..base.len() {
            let mut plus = self.inputs.clone();
            plus.get_mut(&self.wrt).unwrap().as_slice_mut().unwrap()[i] += EPS;
            let mut minus = self.inputs.clone();
            minus.get_mut(&self.wrt).unwrap().as_slice_mut().unwrap()[i] -= EPS;
            let y_p = self.numeric_forward(&backend, &plus);
            let y_m = self.numeric_forward(&backend, &minus);
            grad.as_slice_mut().unwrap()[i] = (y_p - y_m) / (2.0 * EPS);
        }
        grad
    }

    fn numeric_forward(&self, backend: &CpuBackend, data: &HashMap<String, ArrayD<f32>>) -> f32 {
        let ctx = Rc::new(RefCell::new(GraphContext::new()));
        let mut tensors = HashMap::new();
        let mut shapes = HashMap::new();
        for (n, a) in data {
            tensors.insert(n.clone(), Tensor::new_input(&ctx, n));
            shapes.insert(n.clone(), (a.shape().to_vec(), DType::F32));
        }
        let loss = (self.builder)(&tensors);
        let mut g = ctx.borrow().main_graph().clone();
        g.set_output(loss.node_id);
        ShapeInference::run(&mut g, &shapes).unwrap();

        let vals: HashMap<String, Value> = data
            .iter()
            .map(|(k, v)| (k.clone(), Value::Tensor(v.clone())))
            .collect();
        let dev = backend.load_data(&vals).unwrap();
        let mut memo = HashMap::new();
        for (n, t) in &tensors {
            memo.insert((g.id, t.node_id), dev[n].clone());
        }
        let (out, _) = backend.run(&g, memo).unwrap();
        match backend.retrieve_data(&out).unwrap().first().unwrap() {
            Value::Tensor(t) => *t.first().unwrap(),
            _ => panic!("loss is not scalar"),
        }
    }

    /// Запустить тест: сравнить аналитику и числа.
    fn run(&self) {
        let ga = self.analytic();
        let gn = self.numeric();
        assert_close(&ga, &gn, &self.wrt);
    }
}

#[test]
fn grad_layernorm_x() {
    let inputs: HashMap<String, ArrayD<f32>> = [
        ("x", array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        ("gamma", array![[0.5, 1.0, 1.5]]),
        ("beta", array![[0.1, -0.2, 0.3]]),
        ("target", array![[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.clone().into_dyn()))
    .collect();

    GradTest {
        builder: Box::new(|t| {
            let ctx = t.values().next().unwrap().context.clone();
            let x = &t["x"];
            let target = &t["target"];
            let mut ln = LayerNorm::new(&ctx, "ln");
            ln.gamma = t["gamma"].clone();
            ln.beta = t["beta"].clone();
            let output = ln.forward(x);
            // MSE loss: sum((output - target)^2)
            let diff = &output - target;
            let squared = &diff * &diff;
            squared.sum()
        }),
        inputs: inputs.clone(),
        wrt: "x".into(),
    }
    .run();
}

#[test]
fn grad_layernorm_gamma() {
    let inputs: HashMap<String, ArrayD<f32>> = [
        ("x", array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        ("gamma", array![[0.5, 1.0, 1.5]]),
        ("beta", array![[0.1, -0.2, 0.3]]),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.clone().into_dyn()))
    .collect();

    GradTest {
        builder: Box::new(|t| {
            let ctx = t.values().next().unwrap().context.clone();
            let x = &t["x"];
            let mut ln = LayerNorm::new(&ctx, "ln");
            ln.gamma = t["gamma"].clone();
            ln.beta = t["beta"].clone();
            ln.forward(x).sum()
        }),
        inputs,
        wrt: "gamma".into(),
    }
    .run();
}

#[test]
fn grad_layernorm_beta() {
    let inputs: HashMap<String, ArrayD<f32>> = [
        ("x", array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        ("gamma", array![[0.5, 1.0, 1.5]]),
        ("beta", array![[0.1, -0.2, 0.3]]),
    ]
    .iter()
    .map(|(k, v)| (k.to_string(), v.clone().into_dyn()))
    .collect();

    GradTest {
        builder: Box::new(|t| {
            let ctx = t.values().next().unwrap().context.clone();
            let x = &t["x"];
            let mut ln = LayerNorm::new(&ctx, "ln");
            ln.gamma = t["gamma"].clone();
            ln.beta = t["beta"].clone();
            ln.forward(x).sum()
        }),
        inputs,
        wrt: "beta".into(),
    }
    .run();
}

// ============================================================
// Conv2d Autograd Tests
// ============================================================

/// Создание Conv2d через NodeType напрямую в графе
fn create_conv2d(
    ctx: &Rc<RefCell<GraphContext>>,
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
    Tensor { node_id, context: Rc::clone(ctx) }
}

#[test]
fn grad_conv2d_weight() {
    // Input: [1, 1, 4, 4], Weight: [1, 1, 3, 3]
    // Simple 3x3 convolution on 4x4 input
    let input_data = Array4::<f32>::from_shape_fn((1, 1, 4, 4), |(_, _, h, w)| {
        (h * 4 + w) as f32 / 16.0
    }).into_dyn();

    let weight_data = Array4::<f32>::from_shape_fn((1, 1, 3, 3), |(_, _, h, w)| {
        (h * 3 + w + 1) as f32 / 10.0
    }).into_dyn();

    let inputs: HashMap<String, ArrayD<f32>> = [
        ("input".to_string(), input_data),
        ("weight".to_string(), weight_data),
    ].into_iter().collect();

    GradTest {
        builder: Box::new(|t| {
            let ctx = t.values().next().unwrap().context.clone();
            let input = &t["input"];
            let weight = &t["weight"];
            let output = create_conv2d(&ctx, input, weight, (1, 1), (0, 0));
            output.sum()
        }),
        inputs,
        wrt: "weight".into(),
    }
    .run();
}

#[test]
fn grad_conv2d_input() {
    // Input: [1, 1, 4, 4], Weight: [1, 1, 3, 3]
    let input_data = Array4::<f32>::from_shape_fn((1, 1, 4, 4), |(_, _, h, w)| {
        (h * 4 + w) as f32 / 16.0
    }).into_dyn();

    let weight_data = Array4::<f32>::from_shape_fn((1, 1, 3, 3), |(_, _, h, w)| {
        (h * 3 + w + 1) as f32 / 10.0
    }).into_dyn();

    let inputs: HashMap<String, ArrayD<f32>> = [
        ("input".to_string(), input_data),
        ("weight".to_string(), weight_data),
    ].into_iter().collect();

    GradTest {
        builder: Box::new(|t| {
            let ctx = t.values().next().unwrap().context.clone();
            let input = &t["input"];
            let weight = &t["weight"];
            let output = create_conv2d(&ctx, input, weight, (1, 1), (0, 0));
            output.sum()
        }),
        inputs,
        wrt: "input".into(),
    }
    .run();
}

#[test]
fn grad_conv2d_with_padding() {
    // Test convolution with padding
    let input_data = Array4::<f32>::from_shape_fn((1, 1, 4, 4), |(_, _, h, w)| {
        (h * 4 + w + 1) as f32 / 16.0
    }).into_dyn();

    let weight_data = Array4::<f32>::from_shape_fn((1, 1, 3, 3), |(_, _, h, w)| {
        (h * 3 + w + 1) as f32 / 10.0
    }).into_dyn();

    let inputs: HashMap<String, ArrayD<f32>> = [
        ("input".to_string(), input_data),
        ("weight".to_string(), weight_data),
    ].into_iter().collect();

    GradTest {
        builder: Box::new(|t| {
            let ctx = t.values().next().unwrap().context.clone();
            let input = &t["input"];
            let weight = &t["weight"];
            let output = create_conv2d(&ctx, input, weight, (1, 1), (1, 1));
            output.sum()
        }),
        inputs,
        wrt: "weight".into(),
    }
    .run();
}

#[test]
fn grad_conv2d_with_stride() {
    // Test convolution with stride
    let input_data = Array4::<f32>::from_shape_fn((1, 1, 6, 6), |(_, _, h, w)| {
        (h * 6 + w + 1) as f32 / 36.0
    }).into_dyn();

    let weight_data = Array4::<f32>::from_shape_fn((1, 1, 3, 3), |(_, _, h, w)| {
        (h * 3 + w + 1) as f32 / 10.0
    }).into_dyn();

    let inputs: HashMap<String, ArrayD<f32>> = [
        ("input".to_string(), input_data),
        ("weight".to_string(), weight_data),
    ].into_iter().collect();

    GradTest {
        builder: Box::new(|t| {
            let ctx = t.values().next().unwrap().context.clone();
            let input = &t["input"];
            let weight = &t["weight"];
            let output = create_conv2d(&ctx, input, weight, (2, 2), (0, 0));
            output.sum()
        }),
        inputs,
        wrt: "weight".into(),
    }
    .run();
}

#[test]
fn grad_conv2d_multi_channel() {
    // Test multi-channel convolution: input [1, 2, 4, 4], weight [3, 2, 3, 3]
    let input_data = Array4::<f32>::from_shape_fn((1, 2, 4, 4), |(_, c, h, w)| {
        (c * 16 + h * 4 + w + 1) as f32 / 32.0
    }).into_dyn();

    let weight_data = Array4::<f32>::from_shape_fn((3, 2, 3, 3), |(oc, ic, h, w)| {
        (oc * 18 + ic * 9 + h * 3 + w + 1) as f32 / 100.0
    }).into_dyn();

    let inputs: HashMap<String, ArrayD<f32>> = [
        ("input".to_string(), input_data),
        ("weight".to_string(), weight_data),
    ].into_iter().collect();

    GradTest {
        builder: Box::new(|t| {
            let ctx = t.values().next().unwrap().context.clone();
            let input = &t["input"];
            let weight = &t["weight"];
            let output = create_conv2d(&ctx, input, weight, (1, 1), (0, 0));
            output.sum()
        }),
        inputs,
        wrt: "weight".into(),
    }
    .run();
}