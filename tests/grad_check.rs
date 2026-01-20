//  tests/grad_check.rs  (финальная компилирующаяся версия)
//! Проверка корректности обратного прохода численным дифференцированием.

use rustyasg::analysis::shape_inference::ShapeInference;
use rustyasg::asg::{DType, Value};
use rustyasg::autograd::Gradients;
use rustyasg::nn::{LayerNorm, Module};
use rustyasg::runtime::backend::Backend;
use rustyasg::runtime::cpu_backend::CpuBackend;
use rustyasg::tensor::{GraphContext, Tensor};

use ndarray::{array, ArrayD};
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
#[ignore] // TODO: LayerNorm autograd требует архитектурных изменений
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