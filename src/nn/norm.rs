use crate::tensor::{GraphContext, Tensor};
use ndarray::array;
use std::cell::RefCell;
use std::rc::Rc;

use super::module::Module;

#[derive(Debug, Clone)]
pub struct LayerNorm {
    pub gamma: Tensor,
    pub beta: Tensor,
    eps: f32,
}

impl LayerNorm {
    pub fn new(context: &Rc<RefCell<GraphContext>>, name: &str) -> Self {
        LayerNorm {
            gamma: Tensor::new_parameter(context, &format!("{}_gamma", name)),
            beta: Tensor::new_parameter(context, &format!("{}_beta", name)),
            eps: 1e-5,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Нормализация по последней размерности (axis = -1)
        let mean = x.mean_axis(-1);
        let centered = x - &mean;
        let sq = &centered * &centered;
        let var = sq.mean_axis(-1);
        let eps_literal = Tensor::new_literal(&x.context, array![self.eps].into_dyn(), "eps");
        let var_plus_eps = &var + &eps_literal;
        let std = var_plus_eps.sqrt();
        let norm = &centered / &std;
        let scaled = &self.gamma * &norm;
        scaled + &self.beta
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Tensor {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}