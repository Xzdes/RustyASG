//  src/nn/norm.rs
//! Layer-Normalisation для графовой архитектуры с корректным autograd.

use crate::asg::NodeType;
use crate::nn::Module;
use crate::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Малая константа для численной стабильности.
const EPS: f32 = 1e-5;

/// Слой нормализации (LayerNorm) по последней оси.
pub struct LayerNorm {
    pub gamma: Tensor, // обучаемый масштаб
    pub beta: Tensor,  // обучаемый сдвиг
    pub eps: f32,      // константа epsilon
}

impl LayerNorm {
    /// Создаёт новый слой, регистрируя параметры `gamma` и `beta` в графе.
    pub fn new(ctx: &Rc<RefCell<GraphContext>>, name: &str) -> Self {
        let gamma_name = format!("{}.gamma", name);
        let beta_name = format!("{}.beta", name);
        let gamma = Tensor::new_parameter(ctx, &gamma_name);
        let beta = Tensor::new_parameter(ctx, &beta_name);
        Self {
            gamma,
            beta,
            eps: EPS,
        }
    }

    /// Создаёт слой с кастомным epsilon.
    pub fn with_eps(ctx: &Rc<RefCell<GraphContext>>, name: &str, eps: f32) -> Self {
        let mut ln = Self::new(ctx, name);
        ln.eps = eps;
        ln
    }
}

impl Module for LayerNorm {
    /// Прямой проход:  y = gamma * (x - mean) / sqrt(var + eps) + beta
    /// Использует специализированный NodeType::LayerNorm для корректного autograd.
    fn forward(&self, x: &Tensor) -> Tensor {
        // Используем context из входного тензора x
        let ctx = &x.context;
        let node_id = ctx.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::LayerNorm {
                input: x.node_id,
                gamma: self.gamma.node_id,
                beta: self.beta.node_id,
                eps: self.eps,
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
