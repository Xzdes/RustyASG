//! Автоматическое дифференцирование «граф-в-граф» с корректным broadcasting
//! и полной поддержкой Parameter-узлов.

use crate::analysis::shape_inference::{ShapeInference, ShapeInferenceError};
use crate::asg::{Asg, AsgError, NodeId, NodeType, Value};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum AutogradError {
    #[error("ASG error: {0}")]
    Asg(#[from] AsgError),
    #[error("Shape inference: {0}")]
    Shape(#[from] ShapeInferenceError),
}

/// Градиенты, построенные для одного целевого узла.
pub struct Gradients {
    src: Asg,
    grad: Asg,
    map: HashMap<NodeId, NodeId>, // src_node -> grad_node
}

impl Gradients {
    /// Начать построение градиентов для исходного графа `src`.
    pub fn new(src: Asg) -> Self {
        let grad_id = src.id.wrapping_add(1);
        Self {
            src: src.clone(),
            grad: Asg::new(grad_id, Some("grad_graph".into())),
            map: HashMap::new(),
        }
    }

    /// Построить граф градиентов `∂loss/∂wrt`.
    /// Возвращает новый ASG, выходы которого — градиенты по каждому узлу из `wrt`.
    pub fn build(mut self, loss: NodeId, wrt: &[NodeId]) -> Result<Asg, AutogradError> {
        // 1. Топологическая сортировка от loss
        let order = self.topo(loss)?;
        // 2. Создать «1» для ∂loss/∂loss
        let one = self.lit_scalar(1.0);
        self.map.insert(loss, one);

        // 3. Обратный проход для построения основного графа градиентов
        for &node in order.iter().rev() {
            if !self.map.contains_key(&node) {
                continue;
            }
            let g_out = self.map[&node];
            self.backward_node(node, g_out)?;
        }
        
        // 4. Запускаем ShapeInference ОДИН РАЗ, когда граф почти готов.
        let mut initial_shapes = HashMap::new();
        for n in self.src.nodes.values() {
            if let (Some(s), Some(dt)) = (&n.shape, &n.dtype) {
                let ext_name = self.ext_name(n.id);
                initial_shapes.insert(ext_name, (s.clone(), *dt));
            }
        }
        // Устанавливаем временные выходы, чтобы shape inference мог отработать
        self.grad.set_outputs(self.map.values().copied().collect());
        ShapeInference::run(&mut self.grad, &initial_shapes)?;

        // 5. Корректируем градиенты для broadcast-операций, добавляя ReduceSumTo
        for &wrt_id in wrt {
            if let Some(&grad_id) = self.map.get(&wrt_id) {
                let grad_shape = self.grad.get_node(grad_id)?.shape.as_ref().unwrap();
                let param_shape = self.src.get_node(wrt_id)?.shape.as_ref().unwrap();

                if grad_shape != param_shape {
                    let param_as_external = self.import(wrt_id)?;
                    let final_grad = self.grad.add_node(None, NodeType::ReduceSumTo(grad_id, param_as_external));
                    self.map.insert(wrt_id, final_grad);
                }
            }
        }
        
        // 6. Собираем финальные выходы
        let final_outputs: Vec<_> = wrt
            .iter()
            .map(|&n| self.get_or_zero(n))
            .collect::<Result<_, _>>()?;
        self.grad.set_outputs(final_outputs);

        // 7. Запускаем ShapeInference в последний раз, чтобы обработать новые узлы ReduceSumTo
        ShapeInference::run(&mut self.grad, &initial_shapes)?;

        Ok(self.grad)
    }

    // ---------- внутренние вспомогательные методы ----------

    fn ext_name(&self, src_id: NodeId) -> String {
        format!("external_{}_{}", self.src.id, src_id)
    }

    fn lit_scalar(&mut self, v: f32) -> NodeId {
        let arr = ndarray::arr0(v).into_dyn();
        self.grad
            .add_node(None, NodeType::Literal(Value::Tensor(arr)))
    }

    /// Импортировать узел из src как External в grad.
    fn import(&mut self, src_id: NodeId) -> Result<NodeId, AutogradError> {
        let name = self.ext_name(src_id);
        // Проверяем, может уже импортировали узел с таким именем
        if let Some(node) = self.grad.nodes.values().find(|n| n.name.as_deref() == Some(&name)) {
            return Ok(node.id);
        }

        let node = self.src.get_node(src_id)?;
        let new_id = self.grad.add_node(
            Some(name.clone()),
            NodeType::External {
                name,
                source_asg_id: self.src.id,
                source_node_id: src_id,
            },
        );
        {
            let n = self.grad.get_node_mut(new_id)?;
            n.shape = node.shape.clone();
            n.dtype = node.dtype;
        }
        Ok(new_id)
    }

    /// Получить градиент по узлу; если ещё нет — вернуть ноль подходящей формы.
    fn get_or_zero(&mut self, src_id: NodeId) -> Result<NodeId, AutogradError> {
        if let Some(&g) = self.map.get(&src_id) {
            return Ok(g);
        }
        let node = self.src.get_node(src_id)?;
        let shape = node
            .shape
            .as_ref()
            .ok_or_else(|| ShapeInferenceError::MissingInitialShape(
                node.name.clone().unwrap_or_default(),
            ))?;
        let zeros = ndarray::ArrayD::zeros(shape.clone());
        let id = self.grad.add_node(
            Some(format!("zero_grad_{}", src_id)),
            NodeType::Literal(Value::Tensor(zeros)),
        );
        let n = self.grad.get_node_mut(id)?;
        n.shape = Some(shape.clone());
        n.dtype = node.dtype;
        Ok(id)
    }

    /// Простая функция аккумуляции градиента. Проверка на broadcast вынесена.
    fn acc(&mut self, src_id: NodeId, delta: NodeId) -> Result<(), AutogradError> {
        let current_grad = self.map.get(&src_id).copied();
        let new_grad = if let Some(g) = current_grad {
            self.grad.add_node(None, NodeType::Add(g, delta))
        } else {
            delta
        };
        self.map.insert(src_id, new_grad);
        Ok(())
    }

    /// Главный метод backward для одного узла.
    fn backward_node(&mut self, node: NodeId, g_out: NodeId) -> Result<(), AutogradError> {
        let n = self.src.get_node(node)?.clone();
        match &n.node_type {
            // ---------- ЛИСТОВЫЕ УЗЛЫ ----------
            NodeType::Input {..} | NodeType::Parameter { .. } | NodeType::Literal(_) | NodeType::External {..} => {
                // Градиент не течёт дальше.
            }
            
            // ---------- БИНАРНЫЕ ОПЕРАЦИИ ----------
            NodeType::Add(a, b) => {
                self.acc(*a, g_out)?;
                self.acc(*b, g_out)?;
            }
            NodeType::Subtract(a, b) => {
                let minus_one = self.lit_scalar(-1.0);
                let g_b = self.grad.add_node(None, NodeType::Multiply(g_out, minus_one));
                self.acc(*a, g_out)?;
                self.acc(*b, g_b)?;
            }
            NodeType::Multiply(a, b) => {
                let a_node = self.import(*a)?;
                let b_node = self.import(*b)?;
                let g_a = self.grad.add_node(None, NodeType::Multiply(g_out, b_node));
                let g_b = self.grad.add_node(None, NodeType::Multiply(g_out, a_node));
                self.acc(*a, g_a)?;
                self.acc(*b, g_b)?;
            }
            NodeType::Divide(a, b) => {
                let a_node = self.import(*a)?;
                let b_node = self.import(*b)?;
                let g_a = self.grad.add_node(None, NodeType::Divide(g_out, b_node));
                let num = self.grad.add_node(None, NodeType::Multiply(g_out, a_node));
                let b2 = self.grad.add_node(None, NodeType::Multiply(b_node.clone(), b_node));
                let gb_num = self.grad.add_node(None, NodeType::Divide(num, b2));
                let minus_one = self.lit_scalar(-1.0);
                let g_b = self.grad.add_node(None, NodeType::Multiply(gb_num, minus_one));
                self.acc(*a, g_a)?;
                self.acc(*b, g_b)?;
            }
            NodeType::MatrixMultiply(a, b) => {
                let a_node = self.import(*a)?;
                let b_node = self.import(*b)?;
                let a_shape = self.src.get_node(*a)?.shape.as_ref().unwrap();
                let b_shape = self.src.get_node(*b)?.shape.as_ref().unwrap();
                let a_rank = a_shape.len();
                let b_rank = b_shape.len();

                let b_t = self.grad.add_node(None, NodeType::Transpose(b_node, b_rank - 2, b_rank - 1));
                let g_a = self.grad.add_node(None, NodeType::MatrixMultiply(g_out, b_t));

                let a_t = self.grad.add_node(None, NodeType::Transpose(a_node, a_rank - 2, a_rank - 1));
                let g_b = self.grad.add_node(None, NodeType::MatrixMultiply(a_t, g_out));
                
                self.acc(*a, g_a)?;
                self.acc(*b, g_b)?;
            }

            // ---------- УНАРНЫЕ ОПЕРАЦИИ ----------
            NodeType::Mean(x) => {
                let shape = self.src.get_node(*x)?.shape.as_ref().unwrap();
                let n = *shape.last().unwrap_or(&1) as f32;
                let scale = self.lit_scalar(1.0 / n);
                let x_node = self.import(*x)?;
                let g_bcast = self.grad.add_node(None, NodeType::Broadcast(g_out, x_node));
                let g_x = self.grad.add_node(None, NodeType::Multiply(g_bcast, scale));
                self.acc(*x, g_x)?;
            }
            NodeType::Variance(x) => {
                let shape = self.src.get_node(*x)?.shape.as_ref().unwrap();
                let n = *shape.last().unwrap_or(&1) as f32;
                let two_n = self.lit_scalar(2.0 / n);
                let x_node = self.import(*x)?;
                let mean_node = self.src.nodes.values().find(|n| matches!(n.node_type, NodeType::Mean(y) if y == *x)).unwrap();
                let mean = self.import(mean_node.id)?;
                let diff = self.grad.add_node(None, NodeType::Subtract(x_node, mean));
                let local = self.grad.add_node(None, NodeType::Multiply(two_n, diff));
                let g_bcast = self.grad.add_node(None, NodeType::Broadcast(g_out, x_node));
                let g_x = self.grad.add_node(None, NodeType::Multiply(g_bcast, local));
                self.acc(*x, g_x)?;
            }
            NodeType::Sqrt(x) => {
                let sqrt_x = self.import(node)?; // import self (sqrt(x))
                let half = self.lit_scalar(0.5);
                let num = self.grad.add_node(None, NodeType::Multiply(half, g_out));
                let g_x = self.grad.add_node(None, NodeType::Divide(num, sqrt_x));
                self.acc(*x, g_x)?;
            }
            NodeType::ReLU(x) => {
                let x_node = self.import(*x)?;
                let zero = self.lit_scalar(0.0);
                let mask = self.grad.add_node(None, NodeType::GreaterThan(x_node, zero));
                let g_x = self.grad.add_node(None, NodeType::Multiply(g_out, mask));
                self.acc(*x, g_x)?;
            }
            NodeType::Sum(x) => {
                let x_node = self.import(*x)?;
                let g_x = self.grad.add_node(None, NodeType::Broadcast(g_out, x_node));
                self.acc(*x, g_x)?;
            }
            NodeType::Softmax(x) => {
                let s = self.import(node)?; 
                let prod = self.grad.add_node(None, NodeType::Multiply(g_out, s.clone()));
                let sum_prod = self.grad.add_node(None, NodeType::Sum(prod));
                let bcast_sum = self.grad.add_node(None, NodeType::Broadcast(sum_prod, g_out));
                let sub = self.grad.add_node(None, NodeType::Subtract(g_out, bcast_sum));
                let g_x = self.grad.add_node(None, NodeType::Multiply(sub, s));
                self.acc(*x, g_x)?;
            }

            // ---------- ТРАНСФОРМАЦИИ ----------
            NodeType::Transpose(x, ax1, ax2) => {
                let g_x = self.grad.add_node(None, NodeType::Transpose(g_out, *ax1, *ax2));
                self.acc(*x, g_x)?;
            }
            NodeType::Reshape(data, _) => {
                let data_node_src = self.src.get_node(*data)?;
                let original_shape = data_node_src.shape.as_ref().unwrap();

                let shape_data_f32: Vec<f32> = original_shape.iter().map(|&d| d as f32).collect();
                let shape_array = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&[original_shape.len()]), shape_data_f32).unwrap();
                let shape_node_grad = self.grad.add_node(None, NodeType::Literal(Value::Tensor(shape_array)));
                
                let g_x = self.grad.add_node(None, NodeType::Reshape(g_out, shape_node_grad));
                self.acc(*data, g_x)?;
            }

            _ => { /* остальные узлы пока не реализованы */ }
        }
        Ok(())
    }

    fn topo(&self, start: NodeId) -> Result<Vec<NodeId>, AutogradError> {
        let mut vis = HashSet::new();
        let mut order = Vec::new();
        self.dfs(start, &mut vis, &mut order)?;
        Ok(order)
    }

    fn dfs(&self, id: NodeId, vis: &mut HashSet<NodeId>, order: &mut Vec<NodeId>) -> Result<(), AutogradError> {
        if !vis.insert(id) {
            return Ok(());
        }
        let node = self.src.get_node(id)?;
        for inp in inputs_of(&node.node_type) {
            self.dfs(inp, vis, order)?;
        }
        order.push(id);
        Ok(())
    }
}

/// Список всех входных NodeId для данного NodeType.
fn inputs_of(nt: &NodeType) -> Vec<NodeId> {
    match nt {
        NodeType::Add(a, b)
        | NodeType::Subtract(a, b)
        | NodeType::Multiply(a, b)
        | NodeType::Divide(a, b)
        | NodeType::MatrixMultiply(a, b)
        | NodeType::GreaterThan(a, b)
        | NodeType::Power(a, b)
        | NodeType::Broadcast(a, b)
        | NodeType::Reshape(a, b) 
        | NodeType::ReduceSumTo(a, b) => vec![*a, *b],

        NodeType::ReLU(a)
        | NodeType::Sum(a)
        | NodeType::Sigmoid(a)
        | NodeType::Softmax(a)
        | NodeType::Mean(a)
        | NodeType::Variance(a)
        | NodeType::Sqrt(a)
        | NodeType::Log(a)
        | NodeType::Transpose(a, ..) => vec![*a],

        NodeType::MaxPool2d { input, .. } => vec![*input],
        NodeType::MaxUnpool2d { input, original_input, .. } => vec![*input, *original_input],

        _ => vec![],
    }
}