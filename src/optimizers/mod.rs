//! Модуль, содержащий реализации оптимизаторов для обновления весов модели.

use crate::asg::Value;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Трейт, определяющий общий интерфейс для всех оптимизаторов.
pub trait Optimizer {
    /// Выполняет один шаг оптимизации, обновляя веса.
    ///
    /// # Аргументы
    ///
    /// * `parameters` - `HashMap`, содержащий текущие значения весов модели.
    ///   Ключ - это имя параметра (например, "layer1.weights").
    /// * `gradients` - `HashMap`, содержащий вычисленные значения градиентов для этих весов.
    fn step(&self, parameters: &mut HashMap<String, Value>, gradients: &HashMap<String, Value>);
}

/// Реализация оптимизатора Stochastic Gradient Descent (SGD).
pub struct Sgd {
    /// Список символьных тензоров, которые являются обучаемыми параметрами.
    /// Оптимизатор использует их, чтобы знать, какие имена параметров ожидать.
    params: Vec<Tensor>,
    /// Скорость обучения (learning rate).
    lr: f32,
}

impl Sgd {
    /// Создает новый экземпляр оптимизатора SGD.
    ///
    /// # Аргументы
    ///
    /// * `params` - Вектор символьных `Tensor`-дескрипторов, представляющих
    ///   все обучаемые параметры модели.
    /// * `lr` - Скорость обучения.
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self { params, lr }
    }
}

impl Optimizer for Sgd {
    fn step(&self, parameters: &mut HashMap<String, Value>, gradients: &HashMap<String, Value>) {
        // Проходим по всем параметрам, которые отслеживает оптимизатор.
        for param_tensor in &self.params {
            // --- ИСПРАВЛЕНИЕ ОШИБКИ E0716 ---
            // Шаг 1: Заимствуем контекст графа и сохраняем его в `context_guard`.
            // Теперь `context_guard` "владеет" заимствованием и будет жить до конца цикла.
            let context_guard = param_tensor.context.borrow();
            
            // Шаг 2: Получаем ссылку на узел из `context_guard`.
            // `param_node` теперь ссылается на данные внутри `context_guard`, а не на временный объект.
            let param_node = context_guard
                .main_graph()
                .get_node(param_tensor.node_id)
                .expect("Parameter node not found in graph");

            // Шаг 3: Теперь мы можем безопасно использовать `param_node`.
            let param_name = match &param_node.node_type {
                crate::asg::NodeType::Parameter { name } => name,
                _ => panic!("Tensor passed to optimizer is not a parameter"),
            };

            // Получаем текущее значение параметра и его градиент.
            if let (Some(Value::Tensor(param_value)), Some(Value::Tensor(grad_value))) =
                (parameters.get_mut(param_name), gradients.get(param_name))
            {
                // Выполняем обновление весов: param = param - lr * grad
                ndarray::azip!((p in param_value, &g in grad_value) *p = *p - self.lr * g);
            }
        }
    }
}