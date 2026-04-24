//! Weight initialization strategies for neural network parameters.
//!
//! Each `Initializer` knows how to materialize an `ArrayD<f32>` for a given shape.
//! This lets layers declare *how* their parameters should be initialized at
//! construction time, so the framework can populate weights automatically via
//! `GraphContext::init_parameters()` without requiring users to write manual
//! initialization code.

use ndarray::ArrayD;
use ndarray_rand::rand_distr::{Normal, Uniform};
use ndarray_rand::RandomExt;

use crate::asg::Shape;

/// Describes how a parameter should be initialized on first use.
///
/// Used by `ParameterRegistry` to populate weights via
/// `GraphContext::init_parameters()`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Initializer {
    /// All zeros. Default for biases.
    Zeros,
    /// All ones. Default for `LayerNorm` gamma / `BatchNorm` gamma.
    Ones,
    /// Constant value.
    Constant(f32),
    /// Uniform distribution in `[low, high]`.
    Uniform { low: f32, high: f32 },
    /// Normal distribution with given mean and std.
    Normal { mean: f32, std: f32 },
    /// Xavier (Glorot) uniform initialization. Good default for tanh/sigmoid.
    /// Range: `sqrt(6 / (fan_in + fan_out))`.
    XavierUniform,
    /// Xavier (Glorot) normal initialization.
    /// Std: `sqrt(2 / (fan_in + fan_out))`.
    XavierNormal,
    /// Kaiming (He) uniform initialization. Default for ReLU-like activations.
    /// Range: `sqrt(6 / fan_in)`.
    KaimingUniform,
    /// Kaiming (He) normal initialization.
    /// Std: `sqrt(2 / fan_in)`.
    KaimingNormal,
}

impl Initializer {
    /// Materializes this initializer into a concrete tensor of the given shape.
    ///
    /// `fan_in` / `fan_out` are inferred from `shape` using PyTorch conventions:
    /// - 2D `[in, out]` (Linear weights): `fan_in = shape[0]`, `fan_out = shape[1]`.
    /// - 4D `[C_out, C_in, kH, kW]` (Conv weights): `fan_in = C_in * kH * kW`,
    ///   `fan_out = C_out * kH * kW`.
    /// - Otherwise: `fan_in = fan_out = shape.iter().product()`.
    pub fn sample(&self, shape: &Shape) -> ArrayD<f32> {
        let (fan_in, fan_out) = compute_fans(shape);
        let shape_slice: &[usize] = shape.as_slice();

        match *self {
            Initializer::Zeros => ArrayD::zeros(shape_slice),
            Initializer::Ones => ArrayD::ones(shape_slice),
            Initializer::Constant(v) => ArrayD::from_elem(shape_slice, v),
            Initializer::Uniform { low, high } => {
                ArrayD::random(shape_slice, Uniform::new(low, high))
            }
            Initializer::Normal { mean, std } => {
                let dist = Normal::new(mean, std).expect("Normal distribution: std must be >= 0");
                ArrayD::random(shape_slice, dist)
            }
            Initializer::XavierUniform => {
                let bound = (6.0_f32 / (fan_in + fan_out) as f32).sqrt();
                ArrayD::random(shape_slice, Uniform::new(-bound, bound))
            }
            Initializer::XavierNormal => {
                let std = (2.0_f32 / (fan_in + fan_out) as f32).sqrt();
                let dist = Normal::new(0.0, std).expect("Xavier std must be finite");
                ArrayD::random(shape_slice, dist)
            }
            Initializer::KaimingUniform => {
                let bound = (6.0_f32 / fan_in as f32).sqrt();
                ArrayD::random(shape_slice, Uniform::new(-bound, bound))
            }
            Initializer::KaimingNormal => {
                let std = (2.0_f32 / fan_in as f32).sqrt();
                let dist = Normal::new(0.0, std).expect("Kaiming std must be finite");
                ArrayD::random(shape_slice, dist)
            }
        }
    }
}

/// Computes `(fan_in, fan_out)` for common parameter shapes.
fn compute_fans(shape: &Shape) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => {
            let n = shape[0];
            (n, n)
        }
        2 => (shape[0], shape[1]),
        _ => {
            // Conv-like: [C_out, C_in/groups, kH, kW, ...]
            let c_out = shape[0];
            let c_in = shape[1];
            let kernel_size: usize = shape[2..].iter().product();
            let fan_in = c_in * kernel_size.max(1);
            let fan_out = c_out * kernel_size.max(1);
            (fan_in, fan_out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeros_gives_zero_tensor() {
        let arr = Initializer::Zeros.sample(&vec![3, 4]);
        assert_eq!(arr.shape(), &[3, 4]);
        assert!(arr.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn ones_gives_one_tensor() {
        let arr = Initializer::Ones.sample(&vec![5]);
        assert!(arr.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn constant_gives_constant_tensor() {
        let arr = Initializer::Constant(2.5).sample(&vec![2, 2]);
        assert!(arr.iter().all(|&x| x == 2.5));
    }

    #[test]
    fn xavier_uniform_respects_bound() {
        let arr = Initializer::XavierUniform.sample(&vec![100, 100]);
        let bound = (6.0_f32 / 200.0).sqrt();
        assert!(arr.iter().all(|&x| x.abs() <= bound + 1e-6));
    }

    #[test]
    fn kaiming_uniform_has_nonzero_spread() {
        let arr = Initializer::KaimingUniform.sample(&vec![64, 32]);
        let max = arr.iter().cloned().fold(f32::MIN, f32::max);
        let min = arr.iter().cloned().fold(f32::MAX, f32::min);
        assert!(max > 0.0);
        assert!(min < 0.0);
    }

    #[test]
    fn compute_fans_linear() {
        let (fan_in, fan_out) = compute_fans(&vec![128, 64]);
        assert_eq!(fan_in, 128);
        assert_eq!(fan_out, 64);
    }

    #[test]
    fn compute_fans_conv2d() {
        // [C_out=32, C_in=16, kH=3, kW=3]
        let (fan_in, fan_out) = compute_fans(&vec![32, 16, 3, 3]);
        assert_eq!(fan_in, 16 * 9);
        assert_eq!(fan_out, 32 * 9);
    }
}
