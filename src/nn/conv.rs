//! Convolutional layers with declarative shape/init API.

use crate::nn::init::Initializer;
use crate::nn::module::Module;
use crate::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Configuration for Conv2d layer.
#[derive(Debug, Clone)]
pub struct Conv2dConfig {
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels (filters).
    pub out_channels: usize,
    /// Convolution kernel size.
    pub kernel_size: (usize, usize),
    /// Convolution stride.
    pub stride: (usize, usize),
    /// Padding.
    pub padding: (usize, usize),
    /// Dilation (kernel expansion).
    pub dilation: (usize, usize),
    /// Number of groups for grouped/depthwise convolution.
    pub groups: usize,
    /// Use bias.
    pub bias: bool,
}

impl Default for Conv2dConfig {
    fn default() -> Self {
        Self {
            in_channels: 1,
            out_channels: 1,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
            bias: true,
        }
    }
}

impl Conv2dConfig {
    /// Creates Conv2d configuration.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize)) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            ..Default::default()
        }
    }

    /// Sets convolution stride.
    pub fn with_stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Sets padding.
    pub fn with_padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Sets dilation.
    pub fn with_dilation(mut self, dilation: (usize, usize)) -> Self {
        self.dilation = dilation;
        self
    }

    /// Sets number of groups.
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    /// Enables/disables bias.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }
}

/// 2D Convolutional layer.
///
/// Applies 2D convolution to input tensor of shape [N, C_in, H, W].
/// Output tensor has shape [N, C_out, H_out, W_out].
///
/// # Example
///
/// ```rust,ignore
/// use rustyasg::nn::{Conv2d, Module};
///
/// let conv = Conv2d::new(&context, "conv1", 3, 64, (3, 3))
///     .with_padding((1, 1));
/// let output = conv.forward(&input);
/// ```
pub struct Conv2d {
    /// Symbolic descriptor for weight tensor of shape `[C_out, C_in/groups, kH, kW]`.
    pub weight: Tensor,
    /// Optional symbolic descriptor for bias of shape `[C_out]`.
    pub bias: Option<Tensor>,
    /// Layer configuration.
    pub config: Conv2dConfig,
}

impl Conv2d {
    /// Creates a new Conv2d layer with basic parameters.
    ///
    /// # Arguments
    ///
    /// * `context` - Reference to GraphContext
    /// * `name` - Base name for parameters
    /// * `in_channels` - Number of input channels
    /// * `out_channels` - Number of output channels
    /// * `kernel_size` - Kernel size (kH, kW)
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        name: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> Self {
        let config = Conv2dConfig::new(in_channels, out_channels, kernel_size);
        Self::from_config(context, name, config)
    }

    /// Creates Conv2d layer from configuration, registering parameter shapes
    /// and Kaiming-uniform initialization with the graph context.
    pub fn from_config(
        context: &Rc<RefCell<GraphContext>>,
        name: &str,
        config: Conv2dConfig,
    ) -> Self {
        // Weight shape for grouped conv: [C_out, C_in/groups, kH, kW].
        let c_in_per_group = config.in_channels / config.groups;
        let weight_shape = vec![
            config.out_channels,
            c_in_per_group,
            config.kernel_size.0,
            config.kernel_size.1,
        ];

        let weight = Tensor::new_parameter_with_shape(
            context,
            &format!("{}.weight", name),
            weight_shape,
            Initializer::KaimingUniform,
        );

        let bias = if config.bias {
            Some(Tensor::new_parameter_with_shape(
                context,
                &format!("{}.bias", name),
                vec![config.out_channels],
                Initializer::Zeros,
            ))
        } else {
            None
        };

        Self {
            weight,
            bias,
            config,
        }
    }

    /// Sets convolution stride.
    pub fn with_stride(mut self, stride: (usize, usize)) -> Self {
        self.config.stride = stride;
        self
    }

    /// Sets padding.
    pub fn with_padding(mut self, padding: (usize, usize)) -> Self {
        self.config.padding = padding;
        self
    }

    /// Sets dilation.
    pub fn with_dilation(mut self, dilation: (usize, usize)) -> Self {
        self.config.dilation = dilation;
        self
    }

    /// Sets number of groups.
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.config.groups = groups;
        self
    }
}

impl Module for Conv2d {
    /// Applies convolution to input.
    fn forward(&self, inputs: &Tensor) -> Tensor {
        inputs.conv2d(
            &self.weight,
            self.bias.as_ref(),
            self.config.stride,
            self.config.padding,
            self.config.dilation,
            self.config.groups,
        )
    }

    /// Returns trainable parameters of the layer.
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}

/// Transposed 2D convolutional layer (deconvolution).
///
/// Used for increasing spatial dimensions (upsampling),
/// for example in autoencoder decoders and GAN generators.
pub struct ConvTranspose2d {
    /// Weights of shape `[C_in, C_out/groups, kH, kW]`.
    pub weight: Tensor,
    /// Optional bias of shape `[C_out]`.
    pub bias: Option<Tensor>,
    /// Stride.
    pub stride: (usize, usize),
    /// Padding.
    pub padding: (usize, usize),
    /// Output padding.
    pub output_padding: (usize, usize),
    /// Dilation.
    pub dilation: (usize, usize),
    /// Number of groups.
    pub groups: usize,
}

impl ConvTranspose2d {
    /// Creates a new ConvTranspose2d layer, registering parameter shapes and
    /// Kaiming-uniform init with the graph context.
    ///
    /// # Arguments
    /// * `context` — shared graph context.
    /// * `name` — parameter-name prefix.
    /// * `in_channels` — number of input channels.
    /// * `out_channels` — number of output channels.
    /// * `kernel_size` — `(kH, kW)` of the transposed convolution kernel.
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        name: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> Self {
        // ConvTranspose2d weight layout in this framework: [C_in, C_out/groups, kH, kW].
        // We default to groups=1, so the divisor is 1.
        let weight = Tensor::new_parameter_with_shape(
            context,
            &format!("{}.weight", name),
            vec![in_channels, out_channels, kernel_size.0, kernel_size.1],
            Initializer::KaimingUniform,
        );

        let bias = Some(Tensor::new_parameter_with_shape(
            context,
            &format!("{}.bias", name),
            vec![out_channels],
            Initializer::Zeros,
        ));

        Self {
            weight,
            bias,
            stride: (1, 1),
            padding: (0, 0),
            output_padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
        }
    }

    pub fn with_stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    pub fn with_padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    pub fn with_output_padding(mut self, output_padding: (usize, usize)) -> Self {
        self.output_padding = output_padding;
        self
    }

    pub fn without_bias(mut self) -> Self {
        self.bias = None;
        self
    }
}

impl Module for ConvTranspose2d {
    fn forward(&self, inputs: &Tensor) -> Tensor {
        inputs.conv_transpose2d(
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
            self.groups,
        )
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_creation() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let conv = Conv2d::new(&context, "conv1", 3, 64, (3, 3))
            .with_padding((1, 1))
            .with_stride((1, 1));

        assert_eq!(conv.config.in_channels, 3);
        assert_eq!(conv.config.out_channels, 64);
        assert_eq!(conv.config.kernel_size, (3, 3));
        assert_eq!(conv.config.padding, (1, 1));
        assert!(conv.bias.is_some());
    }

    #[test]
    fn test_conv2d_forward() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let input = Tensor::new_input(&context, "input");
        let conv = Conv2d::new(&context, "conv1", 3, 64, (3, 3));

        let _ = conv.forward(&input);

        // Check that graph contains Conv2d operation
        let graph = context.borrow().main_graph().clone();
        assert!(graph.nodes.len() > 2); // input + weight + bias + conv2d
    }

    #[test]
    fn test_conv2d_no_bias() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let config = Conv2dConfig::new(3, 64, (3, 3)).with_bias(false);
        let conv = Conv2d::from_config(&context, "conv1", config);

        assert!(conv.bias.is_none());
        assert_eq!(conv.parameters().len(), 1);
    }

    #[test]
    fn test_conv_transpose2d() {
        let context = Rc::new(RefCell::new(GraphContext::new()));
        let input = Tensor::new_input(&context, "input");
        let deconv = ConvTranspose2d::new(&context, "deconv1", 64, 3, (4, 4))
            .with_stride((2, 2))
            .with_padding((1, 1));

        let _ = deconv.forward(&input);
        assert_eq!(deconv.parameters().len(), 2); // weight + bias
    }
}
