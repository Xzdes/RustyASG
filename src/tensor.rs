//! Модуль, определяющий `Tensor` и `GraphContext`.
//!
//! В новой архитектуре `Tensor` больше не является контейнером для данных.
//! Вместо этого, это легковесный "дескриптор" (handle) или "символьная переменная",
//! которая представляет узел в `Абстрактном Семантическом Графе` (ASG).
//!
//! Все операции над тензорами (`add`, `dot` и т.д.) не выполняют вычисления
//! немедленно, а добавляют соответствующие узлы в граф.
//!
//! `GraphContext` - это центральный объект, который владеет и управляет
//! построением ASG.

use crate::asg::{Asg, NodeId, NodeType, Value};
use ndarray::ArrayD;
use std::cell::RefCell;
use std::ops::{Add, Div, Mul, Sub};
use std::rc::Rc;

/// Контекст, который владеет и управляет построением одного или нескольких ASG.
///
/// Этот объект обернут в `Rc<RefCell<>>`, чтобы его можно было безопасно
/// разделять между множеством `Tensor` дескрипторов.
#[derive(Debug, Clone)]
pub struct GraphContext {
    // Пока что для простоты работаем с одним главным графом.
    // В будущем здесь может быть коллекция графов для поддержки вложенности.
    main_graph: Asg,
}

impl GraphContext {
    /// Создает новый, пустой контекст графа.
    pub fn new() -> Self {
        Self {
            main_graph: Asg::new(0, Some("main".to_string())),
        }
    }

    /// Получает изменяемую ссылку на основной граф для его построения.
    pub fn main_graph_mut(&mut self) -> &mut Asg {
        &mut self.main_graph
    }

    /// Получает иммутабельную ссылку на основной граф.
    pub fn main_graph(&self) -> &Asg {
        &self.main_graph
    }
}

impl Default for GraphContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Символьный дескриптор, представляющий узел в графе вычислений (ASG).
///
/// Этот объект не содержит реальных данных. Он состоит из ID узла и ссылки
/// на `GraphContext`, в котором этот узел существует.
///
/// Любая операция над этим объектом приводит к добавлению нового узла в граф.
#[derive(Debug, Clone)]
pub struct Tensor {
    /// ID узла в ASG, который представляет этот тензор.
    pub node_id: NodeId,
    /// Разделяемая ссылка на контекст, в котором строится граф.
    pub context: Rc<RefCell<GraphContext>>,
}

impl Tensor {
    /// Создает новый "входной" узел в графе и возвращает дескриптор для него.
    /// Входные узлы - это "переменные" графа, в которые будут подаваться
    /// реальные данные во время выполнения.
    pub fn new_input(context: &Rc<RefCell<GraphContext>>, name: &str) -> Self {
        let mut ctx = context.borrow_mut();
        let graph = ctx.main_graph_mut();

        let node_id = graph.add_node(
            Some(name.to_string()),
            NodeType::Input {
                name: name.to_string(),
            },
        );

        // Регистрируем этот узел как один из входов графа.
        graph.inputs.push(node_id);

        Self {
            node_id,
            context: Rc::clone(context),
        }
    }

    /// Создает новый "параметр" в графе.
    /// Параметры - это обучаемые веса модели.
    pub fn new_parameter(context: &Rc<RefCell<GraphContext>>, name: &str) -> Self {
        let node_id = context.borrow_mut().main_graph_mut().add_node(
            Some(name.to_string()),
            NodeType::Parameter {
                name: name.to_string(),
            },
        );
        Self {
            node_id,
            context: Rc::clone(context),
        }
    }

    /// Создает новый узел-константу (литерал) из реальных данных.
    /// Эти данные будут встроены непосредственно в граф.
    pub fn new_literal(context: &Rc<RefCell<GraphContext>>, data: ArrayD<f32>, name: &str) -> Self {
        let node_id = context.borrow_mut().main_graph_mut().add_node(
            Some(name.to_string()),
            NodeType::Literal(Value::Tensor(data)),
        );
        Self {
            node_id,
            context: Rc::clone(context),
        }
    }

    // --- Математические операции ---

    pub fn pow(&self, power: &Tensor) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Power(self.node_id, power.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    pub fn dot(&self, other: &Tensor) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::MatrixMultiply(self.node_id, other.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    pub fn sqrt(&self) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Sqrt(self.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    // --- Функции активации ---

    pub fn relu(&self) -> Self {
        let node_id = self
            .context
            .borrow_mut()
            .main_graph_mut()
            .add_node(None, NodeType::ReLU(self.node_id));
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    pub fn sigmoid(&self) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Sigmoid(self.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }
    
    pub fn softmax(&self) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Softmax(self.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    // --- Операции редукции ---

    pub fn sum(&self) -> Self {
        let node_id = self
            .context
            .borrow_mut()
            .main_graph_mut()
            .add_node(None, NodeType::Sum(self.node_id));
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    pub fn mean(&self) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Mean(self.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }

    pub fn variance(&self) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Variance(self.node_id),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }
    
    // --- Операции трансформации ---

    pub fn reshape(&self, shape: Vec<i64>) -> Self {
        let shape_data = ArrayD::from_shape_vec(ndarray::IxDyn(&[shape.len()]), shape).unwrap();
        let shape_node = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Literal(Value::Tensor(shape_data.mapv(|x| x as f32))),
        );
        let reshape_node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Reshape(self.node_id, shape_node),
        );
        Self {
            node_id: reshape_node_id,
            context: Rc::clone(&self.context),
        }
    }

    pub fn transpose(&self, axis1: usize, axis2: usize) -> Self {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Transpose(self.node_id, axis1, axis2),
        );
        Self {
            node_id,
            context: Rc::clone(&self.context),
        }
    }
}

// Реализация операторов для удобного синтаксиса `a + b`.

impl Add<&Tensor> for &Tensor {
    type Output = Tensor;
    fn add(self, rhs: &Tensor) -> Self::Output {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Add(self.node_id, rhs.node_id),
        );
        Tensor {
            node_id,
            context: Rc::clone(&self.context),
        }
    }
}

impl Sub<&Tensor> for &Tensor {
    type Output = Tensor;
    fn sub(self, rhs: &Tensor) -> Self::Output {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Subtract(self.node_id, rhs.node_id),
        );
        Tensor {
            node_id,
            context: Rc::clone(&self.context),
        }
    }
}

impl Mul<&Tensor> for &Tensor {
    type Output = Tensor;
    fn mul(self, rhs: &Tensor) -> Self::Output {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Multiply(self.node_id, rhs.node_id),
        );
        Tensor {
            node_id,
            context: Rc::clone(&self.context),
        }
    }
}

impl Div<&Tensor> for &Tensor {
    type Output = Tensor;
    fn div(self, rhs: &Tensor) -> Self::Output {
        let node_id = self.context.borrow_mut().main_graph_mut().add_node(
            None,
            NodeType::Divide(self.node_id, rhs.node_id),
        );
        Tensor {
            node_id,
            context: Rc::clone(&self.context),
        }
    }
}