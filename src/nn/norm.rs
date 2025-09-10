//  src/nn/norm.rs  (полностью исправленный)
//! Layer-Normalisation для графовой архитектуры с корректным autograd.

use crate::nn::Module;
use crate::tensor::{GraphContext, Tensor};
use ndarray::arr0;
use std::cell::RefCell;
use std::rc::Rc;

/// Малая константа для численной стабильности.
const EPS: f32 = 1e-5;

/// Слой нормализации (LayerNorm) по последней оси.
pub struct LayerNorm {
    pub gamma: Tensor, // обучаемый масштаб
    pub beta: Tensor,  // обучаемый сдвиг
    eps: Tensor,       // константа-скаляр
}

impl LayerNorm {
    /// Создаёт новый слой, регистрируя параметры `gamma` и `beta` в графе.
    pub fn new(ctx: &Rc<RefCell<GraphContext>>, name: &str) -> Self {
        let gamma_name = format!("{}.gamma", name);
        let beta_name = format!("{}.beta", name);
        let gamma = Tensor::new_parameter(ctx, &gamma_name);
        let beta = Tensor::new_parameter(ctx, &beta_name);
        let eps = Tensor::new_literal(ctx, arr0(EPS).into_dyn(), &format!("{}.eps", name));
        Self { gamma, beta, eps }
    }
}

impl Module for LayerNorm {
    /// Прямой проход:  y = gamma * (x - mean) / sqrt(var + eps) + beta
    fn forward(&self, x: &Tensor) -> Tensor {
        let mean = x.mean();                              // [..., 1]
        let var = x.variance();                           // [..., 1]
        let denom = (&var + &self.eps).sqrt();            // [..., 1]
        let normalized = &(x - &mean) / &denom;           // [..., C]
        let scaled = &normalized * &self.gamma;           // [..., C]
        &scaled + &self.beta                              // [..., C]
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}