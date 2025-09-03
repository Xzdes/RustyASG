//! Модуль для нативной real-time визуализации ASG с помощью egui.

use crate::asg::{Asg, Node, NodeId, NodeType};
use eframe::egui;
use eframe::epaint::Shape;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Topo;
use std::collections::HashMap;
use std::sync::mpsc::Receiver;

const NODE_WIDTH: f32 = 180.0;
const NODE_HEIGHT: f32 = 60.0;
const HORIZONTAL_SPACING: f32 = 80.0;
const VERTICAL_SPACING: f32 = 60.0;

/// Главная структура нашего GUI-приложения.
pub struct GraphViewerApp {
    rx: Receiver<Asg>,
    asg: Option<Asg>,
    node_positions: HashMap<NodeId, egui::Pos2>,
    is_panning: bool,
    pan_offset: egui::Vec2,
}

impl GraphViewerApp {
    pub fn new(cc: &eframe::CreationContext<'_>, rx: Receiver<Asg>) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        Self {
            rx,
            asg: None,
            node_positions: HashMap::new(),
            is_panning: false,
            pan_offset: egui::Vec2::ZERO,
        }
    }

    /// Простой алгоритм иерархического размещения узлов.
    fn simple_layered_layout(&mut self, asg: &Asg) {
        self.node_positions.clear();
        if asg.nodes.is_empty() {
            return;
        }

        let mut graph = DiGraph::<NodeId, ()>::new();
        let mut node_map = HashMap::new();

        for &id in asg.nodes.keys() {
            let index = graph.add_node(id);
            node_map.insert(id, index);
        }

        for (id, node) in &asg.nodes {
            let to_idx = node_map[id];
            for &input_id in &get_node_inputs(&node.node_type) {
                if let Some(&from_idx) = node_map.get(&input_id) {
                    graph.add_edge(from_idx, to_idx, ());
                }
            }
        }
        
        let mut layers: HashMap<usize, Vec<NodeIndex>> = HashMap::new();
        let mut node_layers: HashMap<NodeIndex, usize> = HashMap::new();
        let mut topo = Topo::new(&graph);

        while let Some(nx) = topo.next(&graph) {
            let max_parent_layer = graph.neighbors_directed(nx, petgraph::Direction::Incoming)
                .filter_map(|p_nx| node_layers.get(&p_nx))
                .max()
                .map_or(0, |l| l + 1);

            node_layers.insert(nx, max_parent_layer);
            layers.entry(max_parent_layer).or_default().push(nx);
        }

        for (layer_idx, nodes_in_layer) in layers.iter() {
            let y_pos = *layer_idx as f32 * (NODE_HEIGHT + VERTICAL_SPACING);
            let layer_width = nodes_in_layer.len() as f32 * (NODE_WIDTH + HORIZONTAL_SPACING);
            let start_x = -layer_width / 2.0;

            for (i, &node_idx) in nodes_in_layer.iter().enumerate() {
                let x_pos = start_x + i as f32 * (NODE_WIDTH + HORIZONTAL_SPACING);
                let node_id = graph[node_idx];
                self.node_positions.insert(node_id, egui::pos2(x_pos, y_pos));
            }
        }
    }
}

impl eframe::App for GraphViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if let Ok(new_asg) = self.rx.try_recv() {
            self.simple_layered_layout(&new_asg);
            self.asg = Some(new_asg);
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) =
                ui.allocate_painter(ui.available_size(), egui::Sense::drag());

            // FIX (Warning): `drag_released` заменен на `drag_stopped`
            if response.drag_started() { self.is_panning = true; }
            if response.dragged() && self.is_panning { self.pan_offset += response.drag_delta(); }
            if response.drag_stopped() { self.is_panning = false; }

            if let Some(asg) = &self.asg {
                let center = response.rect.center();

                // Рисуем ребра
                for (id, node) in &asg.nodes {
                    if let Some(pos1) = self.node_positions.get(id) {
                        for &input_id in &get_node_inputs(&node.node_type) {
                            if let Some(pos2) = self.node_positions.get(&input_id) {
                                let p1 = (center.to_vec2() + self.pan_offset + pos1.to_vec2()).to_pos2();
                                let p2 = (center.to_vec2() + self.pan_offset + pos2.to_vec2()).to_pos2();
                                painter.line_segment([p1, p2], egui::Stroke::new(1.5, egui::Color32::GRAY));
                            }
                        }
                    }
                }
                
                // Рисуем узлы
                for (id, node) in &asg.nodes {
                    if let Some(pos) = self.node_positions.get(id) {
                        let node_rect = egui::Rect::from_center_size(
                            (center.to_vec2() + pos.to_vec2() + self.pan_offset).to_pos2(),
                            egui::vec2(NODE_WIDTH, NODE_HEIGHT),
                        );
                        
                        // --- НАЧАЛО ФИНАЛЬНОГО ИСПРАВЛЕНИЯ ---
                        let rounding = egui::Rounding::from(5.0);
                        let (_shape_type, fill_color) = get_node_style(node, asg);
                        let stroke = egui::Stroke::new(1.5, egui::Color32::WHITE);

                        // FIX (Error): Создаем `RectShape`, добавляя все недостающие поля
                        painter.add(Shape::Rect(eframe::epaint::RectShape {
                            rect: node_rect,
                            rounding,
                            fill: fill_color,
                            stroke,
                            // Добавляем недостающие поля с их значениями по умолчанию
                            blur_width: 0.0,
                            fill_texture_id: Default::default(),
                            uv: egui::Rect::NOTHING,
                        }));
                        // --- КОНЕЦ ФИНАЛЬНОГО ИСПРАВЛЕНИЯ ---

                        let label = format_node_label(node);
                        painter.text(
                            node_rect.center(),
                            egui::Align2::CENTER_CENTER,
                            label,
                            egui::FontId::proportional(14.0),
                            egui::Color32::BLACK,
                        );
                    }
                }
            } else {
                ui.label("Ожидание графа для визуализации...");
            }
        });
    }
}

// --- Вспомогательные функции (без изменений) ---

fn format_node_label(node: &Node) -> String {
    let shape_info = node
        .shape
        .as_ref()
        .map_or("".to_string(), |s| format!("\nshape: {:?}", s));

    let type_str = match &node.node_type {
        NodeType::Input { name } => format!("Input\n'{}'", name),
        NodeType::Parameter { name } => format!("Parameter\n'{}'", name),
        NodeType::Literal(_) => "Literal".to_string(),
        NodeType::External { name, .. } => format!("External\n'{}'", name),
        other => format!("{:?}", other).split('(').next().unwrap_or("").to_string(),
    };

    format!("ID: {}\n{}{}", node.id, type_str, shape_info)
}

fn get_node_style(node: &Node, asg: &Asg) -> ((), egui::Color32) {
    let color = if asg.outputs.contains(&node.id) {
        egui::Color32::from_rgb(255, 221, 193) // Персиковый
    } else {
        match &node.node_type {
            NodeType::Input { .. } | NodeType::Parameter { .. } => egui::Color32::from_rgb(194, 239, 235),
            NodeType::External { .. } => egui::Color32::from_rgb(230, 230, 250),
            NodeType::Literal(_) => egui::Color32::from_rgb(224, 224, 224),
            NodeType::Add(..) | NodeType::Subtract(..) | NodeType::Multiply(..) | NodeType::Divide(..) | NodeType::MatrixMultiply(..) => egui::Color32::from_rgb(208, 225, 255),
            NodeType::ReLU(..) | NodeType::Softmax(..) | NodeType::Sigmoid(..) | NodeType::Sqrt(..) => egui::Color32::from_rgb(255, 250, 205),
            NodeType::Sum(..) | NodeType::Mean(..) | NodeType::Variance(..) => egui::Color32::from_rgb(255, 218, 185),
            _ => egui::Color32::WHITE,
        }
    };
    ((), color)
}

fn get_node_inputs(node_type: &NodeType) -> Vec<NodeId> {
    match node_type {
        NodeType::Add(a, b)
        | NodeType::Subtract(a, b)
        | NodeType::Multiply(a, b)
        | NodeType::Divide(a, b)
        | NodeType::MatrixMultiply(a, b)
        | NodeType::GreaterThan(a, b)
        | NodeType::Power(a, b)
        | NodeType::Broadcast(a, b)
        | NodeType::Reshape(a, b) => vec![*a, *b],

        NodeType::ReLU(a)
        | NodeType::Sum(a)
        | NodeType::Sigmoid(a)
        | NodeType::Softmax(a)
        | NodeType::Mean(a)
        | NodeType::Variance(a)
        | NodeType::Sqrt(a)
        | NodeType::Log(a)
        | NodeType::Transpose(a, _, _) => vec![*a],

        NodeType::MaxPool2d { input, .. } => vec![*input],
        NodeType::MaxUnpool2d { input, original_input, .. } => vec![*input, *original_input],
        _ => vec![],
    }
}