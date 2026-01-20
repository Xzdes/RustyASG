// --- Файл: src/nn/conv.rs ---

//! Модуль, реализующий сверточные слои для обработки изображений.

use crate::nn::module::Module;
use crate::tensor::{GraphContext, Tensor};
use std::cell::RefCell;
use std::rc::Rc;

/// Конфигурация для Conv2d слоя.
#[derive(Debug, Clone)]
pub struct Conv2dConfig {
    /// Количество входных каналов.
    pub in_channels: usize,
    /// Количество выходных каналов (фильтров).
    pub out_channels: usize,
    /// Размер ядра свертки.
    pub kernel_size: (usize, usize),
    /// Шаг свертки.
    pub stride: (usize, usize),
    /// Паддинг.
    pub padding: (usize, usize),
    /// Дилатация (расширение ядра).
    pub dilation: (usize, usize),
    /// Количество групп для grouped/depthwise convolution.
    pub groups: usize,
    /// Использовать bias.
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
    /// Создает конфигурацию Conv2d.
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: (usize, usize)) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            ..Default::default()
        }
    }

    /// Устанавливает шаг свертки.
    pub fn with_stride(mut self, stride: (usize, usize)) -> Self {
        self.stride = stride;
        self
    }

    /// Устанавливает паддинг.
    pub fn with_padding(mut self, padding: (usize, usize)) -> Self {
        self.padding = padding;
        self
    }

    /// Устанавливает дилатацию.
    pub fn with_dilation(mut self, dilation: (usize, usize)) -> Self {
        self.dilation = dilation;
        self
    }

    /// Устанавливает количество групп.
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    /// Включает/выключает bias.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }
}

/// 2D Сверточный слой.
///
/// Применяет 2D свертку к входному тензору формы [N, C_in, H, W].
/// Выходной тензор имеет форму [N, C_out, H_out, W_out].
///
/// # Пример
///
/// ```rust,ignore
/// use rustyasg::nn::{Conv2d, Module};
///
/// let conv = Conv2d::new(&context, "conv1", 3, 64, (3, 3))
///     .with_padding((1, 1));
/// let output = conv.forward(&input);
/// ```
pub struct Conv2d {
    /// Символьный дескриптор для тензора весов [C_out, C_in/groups, kH, kW].
    pub weight: Tensor,
    /// Опциональный символьный дескриптор для bias [C_out].
    pub bias: Option<Tensor>,
    /// Конфигурация слоя.
    pub config: Conv2dConfig,
}

impl Conv2d {
    /// Создает новый Conv2d слой с базовыми параметрами.
    ///
    /// # Аргументы
    ///
    /// * `context` - Ссылка на GraphContext
    /// * `name` - Базовое имя для параметров
    /// * `in_channels` - Количество входных каналов
    /// * `out_channels` - Количество выходных каналов
    /// * `kernel_size` - Размер ядра (kH, kW)
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

    /// Создает Conv2d слой из конфигурации.
    pub fn from_config(
        context: &Rc<RefCell<GraphContext>>,
        name: &str,
        config: Conv2dConfig,
    ) -> Self {
        let weight_name = format!("{}.weight", name);
        let weight = Tensor::new_parameter(context, &weight_name);

        let bias = if config.bias {
            let bias_name = format!("{}.bias", name);
            Some(Tensor::new_parameter(context, &bias_name))
        } else {
            None
        };

        Self { weight, bias, config }
    }

    /// Устанавливает шаг свертки.
    pub fn with_stride(mut self, stride: (usize, usize)) -> Self {
        self.config.stride = stride;
        self
    }

    /// Устанавливает паддинг.
    pub fn with_padding(mut self, padding: (usize, usize)) -> Self {
        self.config.padding = padding;
        self
    }

    /// Устанавливает дилатацию.
    pub fn with_dilation(mut self, dilation: (usize, usize)) -> Self {
        self.config.dilation = dilation;
        self
    }

    /// Устанавливает количество групп.
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.config.groups = groups;
        self
    }
}

impl Module for Conv2d {
    /// Применяет свертку к входу.
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

    /// Возвращает обучаемые параметры слоя.
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref bias) = self.bias {
            params.push(bias.clone());
        }
        params
    }
}

/// Транспонированный 2D сверточный слой (деконволюция).
///
/// Используется для увеличения пространственных размеров (upsampling),
/// например в декодерах автоэнкодеров и генераторах GAN.
pub struct ConvTranspose2d {
    /// Веса [C_in, C_out/groups, kH, kW].
    pub weight: Tensor,
    /// Опциональный bias [C_out].
    pub bias: Option<Tensor>,
    /// Шаг.
    pub stride: (usize, usize),
    /// Паддинг.
    pub padding: (usize, usize),
    /// Выходной паддинг.
    pub output_padding: (usize, usize),
    /// Дилатация.
    pub dilation: (usize, usize),
    /// Количество групп.
    pub groups: usize,
}

impl ConvTranspose2d {
    /// Создает новый ConvTranspose2d слой.
    pub fn new(
        context: &Rc<RefCell<GraphContext>>,
        name: &str,
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
    ) -> Self {
        let weight_name = format!("{}.weight", name);
        let weight = Tensor::new_parameter(context, &weight_name);

        let bias_name = format!("{}.bias", name);
        let bias = Some(Tensor::new_parameter(context, &bias_name));

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

        let output = conv.forward(&input);

        // Проверяем, что граф содержит Conv2d операцию
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

        let output = deconv.forward(&input);
        assert_eq!(deconv.parameters().len(), 2); // weight + bias
    }
}
