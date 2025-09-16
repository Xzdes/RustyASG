//! «WGPU» backend: CPU-эмуляция вычислений для совместимости с интерфейсом.
//! Исправления под ndarray 0.16: into_shape_with_order/for_each и сравнения через Zip.

use crate::analysis::shape_inference::ShapeInference;
use crate::asg::{Asg, DType, Node, NodeId, NodeType, Value};
use ndarray::{Array, ArrayD, Axis, IxDyn};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum RuntimeError {
    #[error("Missing input '{0}' for node {1}")]
    MissingInput(String, NodeId),
    #[error("Missing parameter '{0}' for node {1}")]
    MissingParameter(String, NodeId),
    #[error("Shape error: {0}")]
    ShapeError(String),
    #[error("Unsupported op at node {0}")]
    Unsupported(NodeId),
}

pub struct WgpuBackend;

impl WgpuBackend {
    pub fn run(
        main_asg: &mut Asg,
        inputs: &HashMap<String, ArrayD<f32>>,
        params: &HashMap<String, ArrayD<f32>>,
    ) -> Result<HashMap<NodeId, ArrayD<f32>>, RuntimeError> {
        // как и в CPU-бэкенде: подготовим формы и типы
        let initial_shapes = collect_initial_shapes(main_asg, inputs, params);
        ShapeInference::run(main_asg, &initial_shapes)
            .map_err(|e| RuntimeError::ShapeError(format!("{:?}", e)))?;

        let sorted_nodes = ShapeInference::topological_sort(main_asg)
            .map_err(|e| RuntimeError::ShapeError(format!("topological sort failed: {:?}", e)))?;

        let mut values: HashMap<NodeId, ArrayD<f32>> = HashMap::new();

        for id in sorted_nodes {
            let node = main_asg.get_node(id).unwrap();

            match &node.node_type {
                NodeType::Input { name } => {
                    if let Some(arr) = inputs.get(name) {
                        values.insert(id, arr.clone());
                    } else {
                        return Err(RuntimeError::MissingInput(name.clone(), node.id));
                    }
                }

                NodeType::Parameter { name } => {
                    if let Some(arr) = params.get(name) {
                        values.insert(id, arr.clone());
                    } else {
                        // дефолтная инициализация, как в CPU-бэкенде
                        let def = make_default_parameter(node, name);
                        values.insert(id, def);
                    }
                }

                NodeType::Literal(v) => {
                    match v {
                        Value::Tensor(arr) => values.insert(id, arr.clone()),
                        Value::ScalarF32(x) => {
                            let a = Array::from_elem(IxDyn(&[]), *x).into_dyn();
                            values.insert(id, a)
                        }
                        Value::ScalarI32(x) => {
                            let a = Array::from_elem(IxDyn(&[]), *x as f32).into_dyn();
                            values.insert(id, a)
                        }
                        Value::ScalarBool(b) => {
                            let a = Array::from_elem(IxDyn(&[]), if *b { 1.0 } else { 0.0 }).into_dyn();
                            values.insert(id, a)
                        }
                    };
                }

                NodeType::External { .. } => {
                    return Err(RuntimeError::Unsupported(node.id));
                }

                // --- Бинарные ---
                NodeType::Add(a, b) => {
                    let (l, r) = (values.get(a).unwrap(), values.get(b).unwrap());
                    values.insert(id, (l + r).to_owned());
                }
                NodeType::Subtract(a, b) => {
                    let (l, r) = (values.get(a).unwrap(), values.get(b).unwrap());
                    values.insert(id, (l - r).to_owned());
                }
                NodeType::Multiply(a, b) => {
                    let (l, r) = (values.get(a).unwrap(), values.get(b).unwrap());
                    values.insert(id, (l * r).to_owned());
                }
                NodeType::Divide(a, b) => {
                    let (l, r) = (values.get(a).unwrap(), values.get(b).unwrap());
                    values.insert(id, (l / r).to_owned());
                }
                NodeType::Power(a, b) => {
                    let (l, r) = (values.get(a).unwrap(), values.get(b).unwrap());
                    let mut out = l.clone();
                    ndarray::Zip::from(&mut out)
                        .and(r)
                        .for_each(|o, &p| {
                            *o = (*o).powf(p);
                        });
                    values.insert(id, out);
                }
                NodeType::MatrixMultiply(a, b) => {
                    let (l, r) = (values.get(a).unwrap(), values.get(b).unwrap());
                    values.insert(id, op_matmul(l, r));
                }
                NodeType::GreaterThan(a, b) => {
                    let (l, r) = (values.get(a).unwrap(), values.get(b).unwrap());
                    values.insert(id, op_gt(l, r));
                }
                NodeType::Less(a, b) => {
                    let (l, r) = (values.get(a).unwrap(), values.get(b).unwrap());
                    values.insert(id, op_lt(l, r));
                }
                NodeType::Equal(a, b) => {
                    let (l, r) = (values.get(a).unwrap(), values.get(b).unwrap());
                    values.insert(id, op_eq(l, r));
                }

                // --- Унарные ---
                NodeType::Negate(x) => {
                    let a = values.get(x).unwrap();
                    values.insert(id, a.mapv(|v| -v));
                }
                NodeType::Exp(x) => {
                    let a = values.get(x).unwrap();
                    values.insert(id, a.mapv(|v| v.exp()));
                }
                NodeType::Log(x) => {
                    let a = values.get(x).unwrap();
                    values.insert(id, a.mapv(|v| v.ln()));
                }
                NodeType::Sqrt(x) => {
                    let a = values.get(x).unwrap();
                    values.insert(id, a.mapv(|v| v.sqrt()));
                }
                NodeType::ReLU(x) => {
                    let a = values.get(x).unwrap();
                    values.insert(id, a.mapv(|v| if v > 0.0 { v } else { 0.0 }));
                }
                NodeType::Sigmoid(x) => {
                    let a = values.get(x).unwrap();
                    values.insert(id, a.mapv(|v| 1.0 / (1.0 + (-v).exp())));
                }
                NodeType::Softmax(x) => {
                    let a = values.get(x).unwrap();
                    let axis = a.ndim().saturating_sub(1);
                    values.insert(id, op_softmax(a, axis));
                }

                // --- Редукции ---
                NodeType::Sum(x) => {
                    let a = values.get(x).unwrap();
                    let s = a.sum();
                    values.insert(id, Array::from_elem(IxDyn(&[]), s).into_dyn());
                }
                NodeType::Mean(x) => {
                    let a = values.get(x).unwrap();
                    let axis = a.ndim().saturating_sub(1);
                    values.insert(id, op_mean_axis(a, axis as isize));
                }
                NodeType::MeanAxis(x, axis) => {
                    let a = values.get(x).unwrap();
                    values.insert(id, op_mean_axis(a, *axis));
                }
                NodeType::Variance(x) => {
                    let a = values.get(x).unwrap();
                    let axis = a.ndim().saturating_sub(1);
                    let m = op_mean_axis(a, axis as isize);
                    let m_b = broadcast_like(&m, a);
                    let diff = a - &m_b;
                    let sq = diff.mapv(|v| v * v);
                    let var = op_mean_axis(&sq, axis as isize);
                    values.insert(id, var);
                }

                // --- Трансформации ---
                NodeType::Reshape(x, shape_node) => {
                    let a = values.get(x).unwrap();
                    let target_shape = shape_from_node(main_asg.get_node(*shape_node).unwrap(), &values);
                    let reshaped = a
                        .clone()
                        .into_shape_with_order(IxDyn(&target_shape))
                        .unwrap()
                        .into_dyn();
                    values.insert(id, reshaped);
                }
                NodeType::Transpose(x, a1, a2) => {
                    let a = values.get(x).unwrap();
                    let ndim = a.ndim();
                    let mut axes: Vec<usize> = (0..ndim).collect();
                    if *a1 < ndim && *a2 < ndim {
                        axes.swap(*a1, *a2);
                    }
                    let t = a.view().permuted_axes(axes).to_owned();
                    values.insert(id, t);
                }
                NodeType::Broadcast(x, shape_provider) => {
                    let a = values.get(x).unwrap();
                    let target_shape = shape_from_node(main_asg.get_node(*shape_provider).unwrap(), &values);
                    let b = broadcast_to(a, &target_shape);
                    values.insert(id, b);
                }
                NodeType::ReduceSumTo(x, shape_provider) => {
                    let a = values.get(x).unwrap();
                    let target_shape = shape_from_node(main_asg.get_node(*shape_provider).unwrap(), &values);
                    let reduced = reduce_sum_to(a, &target_shape);
                    values.insert(id, reduced);
                }

                NodeType::MaxPool2d { input, kernel_size, stride } => {
                    let a = values.get(input).unwrap();
                    let out = maxpool2d(a, kernel_size.0, kernel_size.1, stride.0, stride.1);
                    values.insert(id, out);
                }
                NodeType::MaxUnpool2d { input, original_input, kernel_size, stride } => {
                    let pooled = values.get(input).unwrap();
                    let orig = main_asg.get_node(*original_input).unwrap();
                    let orig_shape = orig.shape.clone().unwrap_or_else(|| pooled.shape().to_vec());
                    let unpooled = maxunpool2d_like(pooled, &orig_shape, kernel_size.0, kernel_size.1, stride.0, stride.1);
                    values.insert(id, unpooled);
                }

                NodeType::If { .. } | NodeType::ForLoop { .. } => {
                    return Err(RuntimeError::Unsupported(node.id));
                }
            }
        }

        Ok(values)
    }
}

// ======= вспомогательные части — синхронизированы с CPU-бэкендом =======

fn collect_initial_shapes(
    asg: &Asg,
    inputs: &HashMap<String, ArrayD<f32>>,
    params: &HashMap<String, ArrayD<f32>>,
) -> HashMap<String, (Vec<usize>, DType)> {
    let mut map = HashMap::new();
    for (_id, node) in asg.nodes.iter() {
        match &node.node_type {
            NodeType::Input { name } => {
                if let Some(a) = inputs.get(name) {
                    map.insert(name.clone(), (a.shape().to_vec(), DType::F32));
                }
            }
            NodeType::Parameter { name } => {
                if let Some(a) = params.get(name) {
                    map.insert(name.clone(), (a.shape().to_vec(), DType::F32));
                }
            }
            _ => {}
        }
    }
    map
}

fn make_default_parameter(node: &Node, name: &str) -> ArrayD<f32> {
    let want_ones = name.contains("gamma") || name.contains("weight");
    let fill = if want_ones { 1.0 } else { 0.0 };

    if let Some(shape) = node.shape.clone() {
        let size: usize = shape.iter().product();
        let vec = vec![fill; size];
        Array::from_shape_vec(IxDyn(&shape), vec).unwrap().into_dyn()
    } else {
        Array::from_elem(IxDyn(&[]), fill).into_dyn()
    }
}

fn broadcast_like(a: &ArrayD<f32>, like: &ArrayD<f32>) -> ArrayD<f32> {
    let target = like.shape().to_vec();
    broadcast_to(a, &target)
}

fn broadcast_to(a: &ArrayD<f32>, target_shape: &[usize]) -> ArrayD<f32> {
    if a.shape() == target_shape {
        return a.clone();
    }
    let view = a.view();
    let b = view.broadcast(IxDyn(target_shape)).unwrap();
    b.to_owned()
}

fn reduce_sum_to(a: &ArrayD<f32>, target_shape: &[usize]) -> ArrayD<f32> {
    let mut result = a.clone();
    let a_shape = a.shape().to_vec();
    let rank_a = a_shape.len();
    let rank_t = target_shape.len();
    let max_rank = usize::max(rank_a, rank_t);

    let mut a_ext = vec![1; max_rank];
    let mut t_ext = vec![1; max_rank];
    a_ext[(max_rank - rank_a)..].copy_from_slice(&a_shape);
    t_ext[(max_rank - rank_t)..].copy_from_slice(target_shape);

    for axis in 0..max_rank {
        if a_ext[axis] != t_ext[axis] && t_ext[axis] == 1 {
            let mut tmp = result.sum_axis(Axis(axis));
            tmp = tmp.insert_axis(Axis(axis));
            result = tmp.into_dyn();
        }
    }
    result
}

fn op_matmul(a: &ArrayD<f32>, b: &ArrayD<f32>) -> ArrayD<f32> {
    debug_assert_eq!(a.ndim(), 2);
    debug_assert_eq!(b.ndim(), 2);
    let a2 = a.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
    let b2 = b.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
    let c = a2.dot(&b2);
    c.into_dyn()
}

fn op_gt(a: &ArrayD<f32>, b: &ArrayD<f32>) -> ArrayD<f32> {
    let mut out = a.clone();
    ndarray::Zip::from(&mut out)
        .and(a)
        .and(b)
        .for_each(|o, &x, &y| {
            *o = if x > y { 1.0 } else { 0.0 };
        });
    out
}

fn op_lt(a: &ArrayD<f32>, b: &ArrayD<f32>) -> ArrayD<f32> {
    let mut out = a.clone();
    ndarray::Zip::from(&mut out)
        .and(a)
        .and(b)
        .for_each(|o, &x, &y| {
            *o = if x < y { 1.0 } else { 0.0 };
        });
    out
}

fn op_eq(a: &ArrayD<f32>, b: &ArrayD<f32>) -> ArrayD<f32> {
    let mut out = a.clone();
    ndarray::Zip::from(&mut out)
        .and(a)
        .and(b)
        .for_each(|o, &x, &y| {
            *o = if (x - y).abs() <= f32::EPSILON { 1.0 } else { 0.0 };
        });
    out
}

fn op_softmax(a: &ArrayD<f32>, axis_last: usize) -> ArrayD<f32> {
    let mut maxed = a.clone();
    let max_along = a.map_axis(Axis(axis_last), |row| row.fold(f32::NEG_INFINITY, |acc, &v| acc.max(v)));
    let max_b = broadcast_like(&max_along.into_dyn(), a);
    maxed = &maxed - &max_b;

    let exps = maxed.mapv(|v| v.exp());
    let sums = exps.sum_axis(Axis(axis_last)).insert_axis(Axis(axis_last));
    let sums_b = broadcast_like(&sums.into_dyn(), &exps);
    (&exps / &sums_b).to_owned()
}

fn op_mean_axis(a: &ArrayD<f32>, axis: isize) -> ArrayD<f32> {
    let rank = a.ndim();
    let mut ax = if axis < 0 { (rank as isize + axis) as usize } else { axis as usize };
    if ax >= rank { ax = rank - 1; }
    let sum = a.sum_axis(Axis(ax));
    let count = a.shape()[ax] as f32;
    let mean = &sum / count;
    mean.insert_axis(Axis(ax)).into_dyn()
}

// --- MaxPool/Unpool примитивные реализации (для совместимости) ---

fn maxpool2d(a: &ArrayD<f32>, kh: usize, kw: usize, sh: usize, sw: usize) -> ArrayD<f32> {
    let n = a.shape()[0];
    let c = a.shape()[1];
    let h = a.shape()[2];
    let w = a.shape()[3];
    let out_h = (h - kh) / sh + 1;
    let out_w = (w - kw) / sw + 1;
    let mut out = Array::zeros(IxDyn(&[n, c, out_h, out_w])).into_dyn();

    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let hs = oh * sh;
                    let ws = ow * sw;
                    let mut m = f32::NEG_INFINITY;
                    for i in 0..kh {
                        for j in 0..kw {
                            let v = a[[ni, ci, hs + i, ws + j]];
                            if v > m { m = v; }
                        }
                    }
                    out[[ni, ci, oh, ow]] = m;
                }
            }
        }
    }
    out
}

fn maxunpool2d_like(pooled: &ArrayD<f32>, target_shape: &[usize], _kh: usize, _kw: usize, sh: usize, sw: usize) -> ArrayD<f32> {
    let n = target_shape[0];
    let c = target_shape[1];
    let _h = target_shape[2];
    let _w = target_shape[3];
    let mut out = Array::zeros(IxDyn(target_shape)).into_dyn();

    let out_h = pooled.shape()[2];
    let out_w = pooled.shape()[3];
    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let hs = oh * sh;
                    let ws = ow * sw;
                    out[[ni, ci, hs, ws]] = pooled[[ni, ci, oh, ow]];
                }
            }
        }
    }
    out
}

fn shape_from_node(node: &Node, values: &HashMap<NodeId, ArrayD<f32>>) -> Vec<usize> {
    match &node.node_type {
        NodeType::Literal(Value::Tensor(arr)) => arr.shape().to_vec(),
        _ => {
            if let Some(v) = values.get(&node.id) {
                v.shape().to_vec()
            } else {
                node.shape.clone().unwrap_or_default()
            }
        }
    }
}
