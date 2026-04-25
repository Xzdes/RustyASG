//! Batch Normalization with the declarative shape/init API.
//!
//! Forward and backward are implemented as **specialised** ASG nodes
//! (`NodeType::BatchNorm` and friends) rather than as a composition of
//! generic ops, because correct BatchNorm reduces over *every axis except
//! the channel axis* — something the generic `Mean` / `Variance` operators
//! (which reduce over the last axis only) cannot express.
//!
//! `gamma` and `beta` are 1D parameters of length `num_features` that are
//! broadcast across the non-channel axes during forward. The `channel_axis`
//! defaults to `1`, matching the typical `[N, C]` and `[N, C, H, W]`
//! layouts.

use crate::asg::NodeType;
use crate::nn::init::Initializer;
use crate::nn::Module;
use crate::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Default epsilon for numerical stability.
const DEFAULT_EPS: f32 = 1e-5;

/// Default momentum for running-statistics EMA (currently informational —
/// running stats are tracked outside the graph in v0.4).
const DEFAULT_MOMENTUM: f32 = 0.1;

/// Batch Normalization layer.
///
/// Normalises inputs across **batch + spatial** axes, leaving the channel
/// axis intact:
/// `y[..., c, ...] = gamma[c] * (x[..., c, ...] - mean[c]) / sqrt(var[c] + eps) + beta[c]`.
///
/// Works on 2D tensors `[N, C]` (channel axis = 1) and 4D tensors
/// `[N, C, H, W]` (channel axis = 1). The `channel_axis` is configurable
/// for unusual layouts.
///
/// # Limitations (v0.4)
/// Always uses batch statistics — running mean / running variance for
/// inference-mode are tracked manually outside the graph. See ROADMAP for
/// the v0.5 plan to materialise running stats through the graph.
///
/// Forward and backward correctness is verified by hand-computed tests
/// (`batchnorm_forward_matches_hand_calc`, `batchnorm_backward_matches_hand_calc`,
/// `batchnorm_backward_via_autograd`) on the `[N, C]` layout.
pub struct BatchNorm {
    /// Learnable scale of shape `[num_features]`.
    pub gamma: Tensor,
    /// Learnable shift of shape `[num_features]`.
    pub beta: Tensor,
    /// Epsilon for numerical stability.
    pub eps: f32,
    /// Channel axis (default `1` — works for both `[N, C]` and `[N, C, H, W]`).
    pub channel_axis: usize,
    /// EMA momentum for running statistics (informational in v0.4).
    pub momentum: f32,
    /// Training / inference mode flag.
    pub training: bool,
    /// Layer name (used as parameter-name prefix).
    pub name: String,
    /// Number of features (size of the channel axis).
    pub num_features: usize,
}

impl BatchNorm {
    /// Creates a BatchNorm layer over `num_features` channels with
    /// `channel_axis = 1` (works for `[N, C]` and `[N, C, H, W]`).
    pub fn new(ctx: &Rc<RefCell<GraphContext>>, name: &str, num_features: usize) -> Self {
        Self::with_axis(ctx, name, num_features, 1)
    }

    /// Creates a BatchNorm layer with a custom `channel_axis`.
    pub fn with_axis(
        ctx: &Rc<RefCell<GraphContext>>,
        name: &str,
        num_features: usize,
        channel_axis: usize,
    ) -> Self {
        let gamma = Tensor::new_parameter_with_shape(
            ctx,
            &format!("{}.gamma", name),
            vec![num_features],
            Initializer::Ones,
        );
        let beta = Tensor::new_parameter_with_shape(
            ctx,
            &format!("{}.beta", name),
            vec![num_features],
            Initializer::Zeros,
        );

        Self {
            gamma,
            beta,
            eps: DEFAULT_EPS,
            channel_axis,
            momentum: DEFAULT_MOMENTUM,
            training: true,
            name: name.to_string(),
            num_features,
        }
    }

    /// Sets the EMA momentum (informational in v0.4).
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Sets a custom epsilon.
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }
}

impl Module for BatchNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        let ctx = &x.context;
        let node_id = ctx.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::BatchNorm {
                input: x.node_id,
                gamma: self.gamma.node_id,
                beta: self.beta.node_id,
                eps: self.eps,
                channel_axis: self.channel_axis,
            },
        );
        Tensor {
            node_id,
            context: Rc::clone(ctx),
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batchnorm_registers_shapes() {
        let ctx = Rc::new(RefCell::new(GraphContext::new()));
        let _bn = BatchNorm::new(&ctx, "bn1", 32);

        let borrowed = ctx.borrow();
        assert_eq!(
            borrowed.parameter_meta("bn1.gamma").unwrap().shape,
            vec![32]
        );
        assert_eq!(borrowed.parameter_meta("bn1.beta").unwrap().shape, vec![32]);
    }

    #[test]
    fn batchnorm_train_eval_toggle() {
        let ctx = Rc::new(RefCell::new(GraphContext::new()));
        let mut bn = BatchNorm::new(&ctx, "bn1", 16);

        bn.eval();
        assert!(!bn.training);
        bn.train();
        assert!(bn.training);
    }

    #[test]
    fn batchnorm_default_channel_axis_is_one() {
        let ctx = Rc::new(RefCell::new(GraphContext::new()));
        let bn = BatchNorm::new(&ctx, "bn", 8);
        assert_eq!(bn.channel_axis, 1);
    }

    /// Hand-verified backward correctness for a tiny fixture.
    ///
    /// `x = [[1,2,3], [2,3,4], [3,4,5]]`, gamma=ones, beta=zeros, eps=0.
    /// Per channel `c`: mean=c+2, var=2/3, std=√(2/3), inv_std=√(3/2)≈1.2247,
    /// x_norm column = [-√(3/2), 0, +√(3/2)].
    ///
    /// With dy = [[1,0,0],[0,0,0],[0,0,0]] only channel 0 has non-zero gradient.
    /// Plugging into dx_k = (gamma/sigma) * (dy_k - mean(dy) - x_norm_k * mean(dy*x_norm)):
    ///   mean(dy_ch0) = 1/3, mean(dy*xn_ch0) = -√(3/2)/3.
    ///   dx[0,0] = √(3/2) * (1 - 1/3 - (-√(3/2))*(-√(3/2)/3))
    ///           = √(3/2) * (1 - 1/3 - 1/2) = √(3/2) * (1/6) ≈ 0.2041.
    ///   dx[1,0] = √(3/2) * (0 - 1/3 - 0) ≈ -0.4082.
    ///   dx[2,0] = same as dx[0,0] by symmetry ≈ 0.2041.
    #[test]
    fn batchnorm_backward_matches_hand_calc() {
        use crate::analysis::shape_inference::ShapeInference;
        use crate::asg::{DType, Value};
        use crate::runtime::backend::Backend;
        use crate::runtime::cpu_backend::CpuBackend;
        use ndarray::{array, ArrayD};
        use std::collections::HashMap;

        let ctx = Rc::new(RefCell::new(GraphContext::new()));
        let dy = Tensor::new_input(&ctx, "dy");
        let x = Tensor::new_input(&ctx, "x");
        let gamma = Tensor::new_input(&ctx, "gamma");

        let backward_id = ctx.borrow_mut().main_graph_mut().add_node(
            None,
            crate::asg::NodeType::BatchNormBackward {
                grad_output: dy.node_id,
                input: x.node_id,
                gamma: gamma.node_id,
                eps: 0.0,
                channel_axis: 1,
            },
        );
        ctx.borrow_mut().main_graph_mut().set_output(backward_id);

        let mut shapes = HashMap::new();
        shapes.insert("dy".to_string(), (vec![3, 3], DType::F32));
        shapes.insert("x".to_string(), (vec![3, 3], DType::F32));
        shapes.insert("gamma".to_string(), (vec![3], DType::F32));
        let mut g = ctx.borrow().main_graph().clone();
        ShapeInference::run(&mut g, &shapes).unwrap();

        let mut data = HashMap::new();
        data.insert(
            "x".to_string(),
            Value::Tensor(array![[1.0_f32, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]].into_dyn()),
        );
        data.insert(
            "gamma".to_string(),
            Value::Tensor(array![1.0_f32, 1.0, 1.0].into_dyn()),
        );
        data.insert(
            "dy".to_string(),
            Value::Tensor(array![[1.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]].into_dyn()),
        );

        let backend = CpuBackend::new();
        let device = backend.load_data(&data).unwrap();
        let mut memo = HashMap::new();
        for (name, val) in device {
            let nid = g
                .nodes
                .iter()
                .find(|(_, n)| {
                    matches!(&n.node_type,
                        crate::asg::NodeType::Input { name: nn } if nn == &name)
                })
                .map(|(id, _)| *id)
                .unwrap();
            memo.insert((g.id, nid), val);
        }
        let (out, _) = backend.run(&g, memo).unwrap();
        let result = match &backend.retrieve_data(&out).unwrap()[0] {
            Value::Tensor(t) => t.clone(),
            _ => panic!("expected tensor"),
        };

        let expected: ArrayD<f32> = array![
            [0.2041_f32, 0.0, 0.0],
            [-0.4082, 0.0, 0.0],
            [0.2041, 0.0, 0.0],
        ]
        .into_dyn();

        for (i, (a, b)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "BatchNorm backward mismatch at idx {}: got {} expected {}",
                i,
                a,
                b
            );
        }
    }

    /// Same fixture as `batchnorm_backward_matches_hand_calc` but goes through
    /// the **autograd graph**, not by calling `op_batch_norm_backward` directly.
    ///
    /// We construct `loss = (bn * mask).sum()` where `mask = [[1,0,0],...]`,
    /// so that `d(loss)/d(bn) = mask`. This makes `dy` reaching
    /// `BatchNormBackward` equal to `mask` — same as the direct backward
    /// test — and we expect the same `dx[:, 0]`.
    ///
    /// Catches bugs where autograd wires the gradient graph wrongly.
    #[test]
    fn batchnorm_backward_via_autograd() {
        use crate::analysis::shape_inference::ShapeInference;
        use crate::asg::{DType, Value};
        use crate::autograd::Gradients;
        use crate::runtime::backend::Backend;
        use crate::runtime::cpu_backend::CpuBackend;
        use ndarray::{array, ArrayD};
        use std::collections::HashMap;

        let ctx = Rc::new(RefCell::new(GraphContext::new()));
        let x = Tensor::new_input(&ctx, "x");
        let gamma = Tensor::new_input(&ctx, "gamma");
        let beta = Tensor::new_input(&ctx, "beta");
        let mask = Tensor::new_input(&ctx, "mask");

        // Build: loss = (BatchNorm(x) * mask).sum()
        let bn_id = ctx.borrow_mut().main_graph_mut().add_node(
            None,
            crate::asg::NodeType::BatchNorm {
                input: x.node_id,
                gamma: gamma.node_id,
                beta: beta.node_id,
                eps: 0.0,
                channel_axis: 1,
            },
        );
        let bn = Tensor {
            node_id: bn_id,
            context: Rc::clone(&ctx),
        };
        let masked = &bn * &mask;
        let loss = masked.sum();

        let mut shapes = HashMap::new();
        shapes.insert("x".to_string(), (vec![3, 3], DType::F32));
        shapes.insert("gamma".to_string(), (vec![3], DType::F32));
        shapes.insert("beta".to_string(), (vec![3], DType::F32));
        shapes.insert("mask".to_string(), (vec![3, 3], DType::F32));
        let mut fwd = ctx.borrow().main_graph().clone();
        fwd.set_output(loss.node_id);
        ShapeInference::run(&mut fwd, &shapes).unwrap();

        let grad_graph = Gradients::new(fwd.clone())
            .build(loss.node_id, &[x.node_id])
            .expect("grad build");

        let mut data = HashMap::new();
        data.insert(
            "x".to_string(),
            Value::Tensor(array![[1.0_f32, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0]].into_dyn()),
        );
        data.insert(
            "gamma".to_string(),
            Value::Tensor(array![1.0_f32, 1.0, 1.0].into_dyn()),
        );
        data.insert(
            "beta".to_string(),
            Value::Tensor(array![0.0_f32, 0.0, 0.0].into_dyn()),
        );
        data.insert(
            "mask".to_string(),
            Value::Tensor(array![[1.0_f32, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]].into_dyn()),
        );

        let backend = CpuBackend::new();
        let device = backend.load_data(&data).unwrap();
        let mut memo = HashMap::new();
        for (name, val) in device {
            let nid = fwd
                .nodes
                .iter()
                .find(|(_, n)| {
                    matches!(&n.node_type,
                        crate::asg::NodeType::Input { name: nn } if nn == &name)
                })
                .map(|(id, _)| *id)
                .unwrap();
            memo.insert((fwd.id, nid), val);
        }
        let (_, fwd_memo) = backend.run(&fwd, memo).unwrap();
        let (grad_out, _) = backend.run(&grad_graph, fwd_memo).unwrap();
        let result = match &backend.retrieve_data(&grad_out).unwrap()[0] {
            Value::Tensor(t) => t.clone(),
            _ => panic!("expected tensor"),
        };

        // Same expected as the direct test.
        let expected: ArrayD<f32> = array![
            [0.2041_f32, 0.0, 0.0],
            [-0.4082, 0.0, 0.0],
            [0.2041, 0.0, 0.0],
        ]
        .into_dyn();

        eprintln!("autograd dx = {:?}", result.as_slice().unwrap());
        eprintln!("expected dx = {:?}", expected.as_slice().unwrap());

        for (i, (a, b)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "autograd dx mismatch at idx {}: got {} expected {}",
                i,
                a,
                b
            );
        }
    }

    /// Hand-verified forward correctness on a tiny `[N=2, C=3]` fixture.
    ///
    /// For each channel `c`, values `x[:, c] = [c+1, c+4]` give:
    ///   mean = c + 2.5,  var = 2.25,  std = 1.5,  inv_std = 2/3.
    /// With gamma = ones, beta = zeros, the output should be exactly
    /// `[[-1, -1, -1], [+1, +1, +1]]`.
    #[test]
    fn batchnorm_forward_matches_hand_calc() {
        use crate::analysis::shape_inference::ShapeInference;
        use crate::asg::{DType, Value};
        use crate::runtime::backend::Backend;
        use crate::runtime::cpu_backend::CpuBackend;
        use ndarray::{array, ArrayD};
        use std::collections::HashMap;

        let ctx = Rc::new(RefCell::new(GraphContext::new()));
        let x = Tensor::new_input(&ctx, "x");
        let gamma = Tensor::new_input(&ctx, "gamma");
        let beta = Tensor::new_input(&ctx, "beta");

        let bn_id = ctx.borrow_mut().main_graph_mut().add_node(
            None,
            crate::asg::NodeType::BatchNorm {
                input: x.node_id,
                gamma: gamma.node_id,
                beta: beta.node_id,
                eps: 0.0,
                channel_axis: 1,
            },
        );
        ctx.borrow_mut().main_graph_mut().set_output(bn_id);

        let mut shapes = HashMap::new();
        shapes.insert("x".to_string(), (vec![2, 3], DType::F32));
        shapes.insert("gamma".to_string(), (vec![3], DType::F32));
        shapes.insert("beta".to_string(), (vec![3], DType::F32));
        let mut g = ctx.borrow().main_graph().clone();
        ShapeInference::run(&mut g, &shapes).unwrap();

        let x_data: ArrayD<f32> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        let gamma_data: ArrayD<f32> = array![1.0, 1.0, 1.0].into_dyn();
        let beta_data: ArrayD<f32> = array![0.0, 0.0, 0.0].into_dyn();

        let mut data = HashMap::new();
        data.insert("x".to_string(), Value::Tensor(x_data));
        data.insert("gamma".to_string(), Value::Tensor(gamma_data));
        data.insert("beta".to_string(), Value::Tensor(beta_data));

        let backend = CpuBackend::new();
        let device = backend.load_data(&data).unwrap();
        let mut memo = HashMap::new();
        for (name, val) in device {
            let nid = g
                .nodes
                .iter()
                .find(|(_, n)| {
                    matches!(&n.node_type,
                        crate::asg::NodeType::Input { name: nn } if nn == &name)
                })
                .map(|(id, _)| *id)
                .unwrap();
            memo.insert((g.id, nid), val);
        }
        let (out, _) = backend.run(&g, memo).unwrap();
        let result = match &backend.retrieve_data(&out).unwrap()[0] {
            Value::Tensor(t) => t.clone(),
            _ => panic!("expected tensor"),
        };

        let expected = array![[-1.0_f32, -1.0, -1.0], [1.0, 1.0, 1.0]].into_dyn();
        for (a, b) in result.iter().zip(expected.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "BatchNorm forward mismatch: got {} expected {}",
                a,
                b
            );
        }
    }
}
