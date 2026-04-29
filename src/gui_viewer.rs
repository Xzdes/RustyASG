//! Native real-time ASG visualization powered by `egui`.
//!
//! Phase A of the **Interactive Model Lab** roadmap (read-only inspection),
//! plus the **A++ educational expansion**:
//!
//! - Click any node and the right-side panel explains, in plain English (or
//!   Russian), *what* the operation does, the *formula* behind it, *why* it
//!   shows up in real models, and — for parameters — *what role* it plays
//!   in this specific model (γ/β of LayerNorm, Q/K/V projections, etc.).
//! - Selected node is rendered with a highlighted border, and every edge
//!   touching it is drawn brighter so the user can trace dataflow.
//! - Per-category color coding: parameters / inputs / literals /
//!   activations / arithmetic / reductions / output.
//! - Live loss chart at the bottom of the window, updating as the
//!   compute thread reports each epoch.
//! - Two-language UI (English / Russian) selectable at startup via the
//!   `--lang` flag.
//! - Technical fields (id / name / type / shape / dtype / inputs) are
//!   collapsed under a "Technical details" header at the bottom — primary
//!   focus stays on the explanation.
//!
//! Future phases (B-E) will turn the inspection panel into an editor —
//! see `ROADMAP.md`.

use eframe::egui;
use eframe::epaint::Shape;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::Topo;
use rustyasg::asg::{Asg, Node, NodeId, NodeType};
use std::collections::HashMap;
use std::sync::mpsc::Receiver;

/// Rich, language-aware description of a single ASG node. Used to render
/// the educational sections of the right-side inspector panel.
struct NodeDescription {
    /// One-line headline shown bold at the top (e.g. "Matrix multiplication").
    headline: String,
    /// "What does it do" — the operation in plain English.
    what: String,
    /// Optional formula (rendered in monospace).
    formula: Option<String>,
    /// "Why is it used" — the role of this op in a typical model.
    why: String,
    /// Optional context: where in *this* model the node sits (filled in for
    /// `Parameter` nodes when we recognise the suffix).
    context: Option<String>,
}

const NODE_WIDTH: f32 = 180.0;
const NODE_HEIGHT: f32 = 60.0;
const HORIZONTAL_SPACING: f32 = 80.0;
const VERTICAL_SPACING: f32 = 60.0;

// ============================================================
// Internationalization
// ============================================================

/// UI language. Selected at startup via `--lang en|ru`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Lang {
    En,
    Ru,
}

impl Lang {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "en" | "english" => Some(Lang::En),
            "ru" | "russian" => Some(Lang::Ru),
            _ => None,
        }
    }
}

/// Lookup table of UI strings. English is the primary; Russian mirrors it.
///
/// Unknown keys return a literal `"?"` so a missing translation is visible
/// rather than silently fallback-rendering the key. Add new entries here
/// when introducing new UI strings.
fn tr(lang: Lang, key: &str) -> &'static str {
    let (en, ru): (&'static str, &'static str) = match key {
        "waiting" => (
            "Waiting for a graph to visualize…",
            "Ожидание графа для визуализации…",
        ),
        "inspector_title" => ("Node inspector", "Инспектор узла"),
        "no_selection" => (
            "Click any node to inspect it.",
            "Кликните на любой узел, чтобы рассмотреть его.",
        ),
        "field_id" => ("ID", "ID"),
        "field_name" => ("Name", "Имя"),
        "field_type" => ("Type", "Тип"),
        "field_shape" => ("Shape", "Форма"),
        "field_dtype" => ("Data type", "Тип данных"),
        "field_inputs" => ("Inputs", "Входы"),
        "field_is_output" => ("Graph output", "Выход графа"),
        "field_yes" => ("yes", "да"),
        "field_no" => ("no", "нет"),
        "field_unknown" => ("(unknown)", "(неизвестно)"),
        "loss_chart" => ("Loss", "Потери"),
        "epoch" => ("Epoch", "Эпоха"),
        "no_loss_yet" => (
            "Loss chart will appear once training starts.",
            "График появится, когда начнётся обучение.",
        ),
        "hint_drag" => (
            "Hold left mouse button to pan the canvas.",
            "Зажмите ЛКМ, чтобы перемещать холст.",
        ),
        "category_input" => ("Input", "Вход"),
        "category_param" => ("Parameter", "Параметр"),
        "category_literal" => ("Literal", "Константа"),
        "category_external" => ("External", "Внешний"),
        "category_activation" => ("Activation", "Активация"),
        "category_arithmetic" => ("Arithmetic", "Арифметика"),
        "category_reduction" => ("Reduction", "Редукция"),
        "category_norm" => ("Normalisation", "Нормализация"),
        "category_conv" => ("Convolution", "Свёртка"),
        "category_pool" => ("Pooling", "Пулинг"),
        "category_shape_op" => ("Shape op", "Изменение формы"),
        "category_grad" => ("Gradient", "Градиент"),
        "category_other" => ("Other", "Другое"),

        // Description section headers.
        "section_what" => ("What this node does", "Что делает этот узел"),
        "section_formula" => ("Formula", "Формула"),
        "section_why" => ("Why it's used", "Зачем нужен"),
        "section_context" => ("Role in this model", "Роль в этой модели"),
        "section_technical" => ("Technical details", "Технические данные"),

        // Param-role hints used in `Context` for `Parameter` nodes.
        "role_layernorm_gamma" => (
            "Learnable scale (γ) of LayerNorm. Initialised to ones; learned during training. \
             Re-introduces the per-feature scale that normalisation removed.",
            "Обучаемый масштаб (γ) для LayerNorm. Инициализируется единицами, обучается. \
             Возвращает поканальный масштаб, который убрала нормализация.",
        ),
        "role_layernorm_beta" => (
            "Learnable shift (β) of LayerNorm. Initialised to zeros; learned during training. \
             Lets the network re-introduce a non-zero mean per feature.",
            "Обучаемый сдвиг (β) для LayerNorm. Инициализируется нулями, обучается. \
             Позволяет сети вернуть ненулевое среднее по каналам.",
        ),
        "role_batchnorm_gamma" => (
            "Per-channel scale (γ) of BatchNorm. Initialised to ones; learned. \
             One value per channel, broadcast across batch and spatial axes.",
            "Поканальный масштаб (γ) для BatchNorm. Инициализируется единицами, обучается. \
             Один скаляр на канал, broadcast'ится на batch и spatial оси.",
        ),
        "role_batchnorm_beta" => (
            "Per-channel shift (β) of BatchNorm. Initialised to zeros; learned. \
             One value per channel, broadcast across batch and spatial axes.",
            "Поканальный сдвиг (β) для BatchNorm. Инициализируется нулями, обучается. \
             Один скаляр на канал, broadcast'ится на batch и spatial оси.",
        ),
        "role_linear_weights" => (
            "Weight matrix W of a fully-connected layer. Maps an `in_features`-dim \
             vector to `out_features`-dim. Initialised with Xavier-uniform.",
            "Матрица весов W полносвязного слоя. Превращает вектор размерности `in_features` \
             в вектор `out_features`. Инициализация — Xavier-uniform.",
        ),
        "role_bias" => (
            "Bias vector b. Added to every output position. Initialised to zeros. \
             Provides a per-output offset that doesn't depend on input.",
            "Вектор смещений b. Добавляется к каждому выходу. Инициализация — нули. \
             Даёт поканальное смещение, не зависящее от входа.",
        ),
        "role_conv_weight" => (
            "Convolution kernel of shape `[C_out, C_in/groups, kH, kW]`. \
             Initialised with Kaiming-uniform (suited for ReLU-like activations).",
            "Свёрточное ядро формы `[C_out, C_in/groups, kH, kW]`. \
             Инициализация — Kaiming-uniform (хорошо подходит для ReLU-подобных активаций).",
        ),
        "role_embedding_weight" => (
            "Embedding table: each row is the dense vector for one vocabulary item. \
             Shape `[num_embeddings, embedding_dim]`. Initialised with Normal(0, 0.02) \
             — the standard for transformer-era models (BERT/GPT).",
            "Таблица эмбеддингов: каждая строка — плотный вектор для одного токена словаря. \
             Форма `[num_embeddings, embedding_dim]`. Инициализация — Normal(0, 0.02), \
             как в BERT/GPT.",
        ),
        "role_attn_query" => (
            "Query projection of Multi-Head Attention. Maps each token to a query vector \
             that decides *which* tokens to attend to.",
            "Проекция Query в Multi-Head Attention. Превращает каждый токен в Q-вектор, \
             определяющий, *на какие* токены обращать внимание.",
        ),
        "role_attn_key" => (
            "Key projection of Multi-Head Attention. Each token's key answers *how relevant \
             am I* to incoming queries.",
            "Проекция Key в Multi-Head Attention. Ключ каждого токена отвечает на вопрос \
             «насколько я *релевантен* приходящим Q-запросам».",
        ),
        "role_attn_value" => (
            "Value projection of Multi-Head Attention. Carries the actual content that \
             attention will mix according to query·key scores.",
            "Проекция Value в Multi-Head Attention. Несёт собственно содержание, \
             которое attention смешивает по весам softmax(Q·Kᵀ).",
        ),
        "role_attn_output" => (
            "Output projection of Multi-Head Attention. Mixes the per-head outputs back \
             into a single embedding-dim vector.",
            "Выходная проекция Multi-Head Attention. Смешивает выходы голов обратно \
             в один вектор embedding-dim.",
        ),
        _ => ("?", "?"),
    };
    match lang {
        Lang::En => en,
        Lang::Ru => ru,
    }
}

// ============================================================
// Compute thread → GUI update protocol
// ============================================================

/// Updates the compute thread sends to the GUI.
///
/// The visualiser stays passive: it waits for these messages on a
/// `mpsc::Receiver` and re-renders when one arrives. Future phases will
/// add a reverse `GuiCommand` channel for mutations.
#[derive(Debug, Clone)]
pub enum ComputeUpdate {
    /// Initial graph (or a fully replaced graph) ready for display.
    GraphReady(Asg),
    /// One training step completed.
    EpochDone { epoch: usize, loss: f32 },
}

// ============================================================
// Application state
// ============================================================

/// The main GUI application struct.
pub struct GraphViewerApp {
    rx: Receiver<ComputeUpdate>,
    asg: Option<Asg>,
    node_positions: HashMap<NodeId, egui::Pos2>,
    is_panning: bool,
    pan_offset: egui::Vec2,

    /// ID of the currently selected node, if any.
    selected_node: Option<NodeId>,
    /// Per-epoch loss history for the bottom chart.
    loss_history: Vec<(usize, f32)>,
    /// UI language.
    lang: Lang,
}

impl GraphViewerApp {
    pub fn new(cc: &eframe::CreationContext<'_>, rx: Receiver<ComputeUpdate>, lang: Lang) -> Self {
        cc.egui_ctx.set_visuals(egui::Visuals::dark());
        Self {
            rx,
            asg: None,
            node_positions: HashMap::new(),
            is_panning: false,
            pan_offset: egui::Vec2::ZERO,
            selected_node: None,
            loss_history: Vec::new(),
            lang,
        }
    }

    /// Simple hierarchical node-layout algorithm.
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
            let max_parent_layer = graph
                .neighbors_directed(nx, petgraph::Direction::Incoming)
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
                self.node_positions
                    .insert(node_id, egui::pos2(x_pos, y_pos));
            }
        }
    }

    // --- Side panel rendering ---

    fn render_inspector_panel(&self, ui: &mut egui::Ui) {
        ui.heading(tr(self.lang, "inspector_title"));
        ui.separator();

        let asg = match &self.asg {
            Some(a) => a,
            None => {
                ui.label(tr(self.lang, "waiting"));
                return;
            }
        };

        let selected_id = match self.selected_node {
            Some(id) => id,
            None => {
                ui.label(tr(self.lang, "no_selection"));
                ui.add_space(8.0);
                ui.label(egui::RichText::new(tr(self.lang, "hint_drag")).weak());
                return;
            }
        };

        let node = match asg.nodes.get(&selected_id) {
            Some(n) => n,
            None => {
                ui.label(tr(self.lang, "field_unknown"));
                return;
            }
        };

        // ----- Educational sections -----
        let desc = describe_node(node, asg, self.lang);

        // Headline (bold, slightly larger).
        ui.label(
            egui::RichText::new(&desc.headline)
                .heading()
                .color(egui::Color32::from_rgb(220, 230, 255)),
        );
        ui.add_space(2.0);
        ui.label(
            egui::RichText::new(format!(
                "#{}  ·  {}",
                node.id,
                human_node_type(&node.node_type)
            ))
            .weak(),
        );
        ui.add_space(8.0);

        // What does it do.
        ui.label(
            egui::RichText::new(tr(self.lang, "section_what"))
                .strong()
                .color(egui::Color32::from_rgb(180, 220, 180)),
        );
        ui.label(&desc.what);
        ui.add_space(8.0);

        // Formula (if any).
        if let Some(formula) = &desc.formula {
            ui.label(
                egui::RichText::new(tr(self.lang, "section_formula"))
                    .strong()
                    .color(egui::Color32::from_rgb(180, 220, 180)),
            );
            ui.label(egui::RichText::new(formula).monospace());
            ui.add_space(8.0);
        }

        // Why.
        ui.label(
            egui::RichText::new(tr(self.lang, "section_why"))
                .strong()
                .color(egui::Color32::from_rgb(180, 220, 180)),
        );
        ui.label(&desc.why);
        ui.add_space(8.0);

        // Context (parameter role / output marker).
        if let Some(context) = &desc.context {
            ui.label(
                egui::RichText::new(tr(self.lang, "section_context"))
                    .strong()
                    .color(egui::Color32::from_rgb(255, 220, 130)),
            );
            ui.label(context);
            ui.add_space(8.0);
        }

        // ----- Technical details (collapsible) -----
        ui.separator();
        egui::CollapsingHeader::new(tr(self.lang, "section_technical"))
            .default_open(true)
            .show(ui, |ui| {
                egui::Grid::new("node_inspector_grid")
                    .num_columns(2)
                    .spacing([20.0, 6.0])
                    .show(ui, |ui| {
                        ui.label(egui::RichText::new(tr(self.lang, "field_id")).strong());
                        ui.label(format!("{}", node.id));
                        ui.end_row();

                        if let Some(name) = &node.name {
                            ui.label(egui::RichText::new(tr(self.lang, "field_name")).strong());
                            ui.label(name);
                            ui.end_row();
                        }

                        ui.label(egui::RichText::new(tr(self.lang, "field_type")).strong());
                        ui.label(human_node_type(&node.node_type));
                        ui.end_row();

                        ui.label(egui::RichText::new(tr(self.lang, "field_shape")).strong());
                        ui.label(match &node.shape {
                            Some(s) => format!("{:?}", s),
                            None => tr(self.lang, "field_unknown").to_string(),
                        });
                        ui.end_row();

                        ui.label(egui::RichText::new(tr(self.lang, "field_dtype")).strong());
                        ui.label(match &node.dtype {
                            Some(d) => format!("{:?}", d),
                            None => tr(self.lang, "field_unknown").to_string(),
                        });
                        ui.end_row();

                        ui.label(egui::RichText::new(tr(self.lang, "field_is_output")).strong());
                        ui.label(if asg.outputs.contains(&node.id) {
                            tr(self.lang, "field_yes")
                        } else {
                            tr(self.lang, "field_no")
                        });
                        ui.end_row();
                    });

                // Inputs list.
                let inputs = get_node_inputs(&node.node_type);
                if !inputs.is_empty() {
                    ui.add_space(8.0);
                    ui.label(egui::RichText::new(tr(self.lang, "field_inputs")).strong());
                    for input_id in inputs {
                        let input_label = match asg.nodes.get(&input_id) {
                            Some(n) => format!(
                                "  • #{}  {}",
                                n.id,
                                n.name
                                    .clone()
                                    .unwrap_or_else(|| human_node_type(&n.node_type))
                            ),
                            None => format!("  • #{}  ?", input_id),
                        };
                        ui.label(input_label);
                    }
                }
            });
    }

    // --- Loss chart rendering ---

    fn render_loss_chart(&self, ui: &mut egui::Ui) {
        ui.heading(tr(self.lang, "loss_chart"));

        if self.loss_history.is_empty() {
            ui.label(egui::RichText::new(tr(self.lang, "no_loss_yet")).weak());
            return;
        }

        let last = self.loss_history.last().unwrap();
        ui.label(format!(
            "{} {} = {:.6}",
            tr(self.lang, "epoch"),
            last.0,
            last.1
        ));

        // Manual line chart with `egui::Painter`.
        let desired_size = egui::vec2(ui.available_width(), 120.0);
        let (rect, _) = ui.allocate_exact_size(desired_size, egui::Sense::hover());
        let painter = ui.painter_at(rect);

        // Light frame.
        painter.rect_stroke(
            rect,
            egui::Rounding::ZERO,
            egui::Stroke::new(1.0, egui::Color32::from_gray(80)),
        );

        let n = self.loss_history.len();
        if n < 2 {
            return;
        }

        let min_loss = self
            .loss_history
            .iter()
            .map(|&(_, l)| l)
            .fold(f32::INFINITY, f32::min);
        let max_loss = self
            .loss_history
            .iter()
            .map(|&(_, l)| l)
            .fold(f32::NEG_INFINITY, f32::max);
        let range = (max_loss - min_loss).max(1e-6);

        let pad_left = 8.0;
        let pad_right = 8.0;
        let pad_top = 8.0;
        let pad_bottom = 8.0;
        let plot_w = rect.width() - pad_left - pad_right;
        let plot_h = rect.height() - pad_top - pad_bottom;

        // Map (epoch_idx, loss) → screen point.
        let to_screen = |i: usize, loss: f32| -> egui::Pos2 {
            let x = rect.left() + pad_left + (i as f32 / (n - 1) as f32) * plot_w;
            // `loss = max → top of plot`, `loss = min → bottom`.
            let y = rect.bottom() - pad_bottom - ((loss - min_loss) / range) * plot_h;
            egui::pos2(x, y)
        };

        let stroke = egui::Stroke::new(2.0, egui::Color32::from_rgb(120, 200, 255));
        for w in self.loss_history.windows(2).enumerate() {
            let (i, pair) = w;
            let p1 = to_screen(i, pair[0].1);
            let p2 = to_screen(i + 1, pair[1].1);
            painter.line_segment([p1, p2], stroke);
        }

        // Min / max labels in the corners.
        painter.text(
            egui::pos2(rect.left() + 4.0, rect.top() + 4.0),
            egui::Align2::LEFT_TOP,
            format!("max: {:.4}", max_loss),
            egui::FontId::proportional(11.0),
            egui::Color32::from_gray(180),
        );
        painter.text(
            egui::pos2(rect.left() + 4.0, rect.bottom() - 4.0),
            egui::Align2::LEFT_BOTTOM,
            format!("min: {:.4}", min_loss),
            egui::FontId::proportional(11.0),
            egui::Color32::from_gray(180),
        );
    }
}

impl eframe::App for GraphViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Drain pending updates from the compute thread (non-blocking).
        while let Ok(update) = self.rx.try_recv() {
            match update {
                ComputeUpdate::GraphReady(new_asg) => {
                    self.simple_layered_layout(&new_asg);
                    self.asg = Some(new_asg);
                    self.selected_node = None;
                }
                ComputeUpdate::EpochDone { epoch, loss } => {
                    self.loss_history.push((epoch, loss));
                }
            }
        }

        // Redraw periodically so the loss chart updates even without input.
        ctx.request_repaint_after(std::time::Duration::from_millis(100));

        // Right side panel: node inspector.
        egui::SidePanel::right("inspector_panel")
            .resizable(true)
            .default_width(300.0)
            .min_width(220.0)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.render_inspector_panel(ui);
                });
            });

        // Bottom panel: loss chart.
        egui::TopBottomPanel::bottom("loss_panel")
            .resizable(true)
            .default_height(160.0)
            .min_height(80.0)
            .show(ctx, |ui| {
                self.render_loss_chart(ui);
            });

        // Central panel: graph canvas.
        egui::CentralPanel::default().show(ctx, |ui| {
            let canvas_response =
                ui.allocate_response(ui.available_size(), egui::Sense::click_and_drag());
            let canvas_rect = canvas_response.rect;
            let painter = ui.painter_at(canvas_rect);

            // Pan with left-mouse drag (only when nothing is hit).
            if canvas_response.drag_started() {
                self.is_panning = true;
            }
            if canvas_response.dragged() && self.is_panning {
                self.pan_offset += canvas_response.drag_delta();
            }
            if canvas_response.drag_stopped() {
                self.is_panning = false;
            }

            let asg = match &self.asg {
                Some(a) => a.clone(),
                None => {
                    ui.label(tr(self.lang, "waiting"));
                    return;
                }
            };

            let center = canvas_rect.center();

            // First pass: edges.
            for (id, node) in &asg.nodes {
                if let Some(pos1) = self.node_positions.get(id) {
                    for &input_id in &get_node_inputs(&node.node_type) {
                        if let Some(pos2) = self.node_positions.get(&input_id) {
                            let p1 =
                                (center.to_vec2() + self.pan_offset + pos1.to_vec2()).to_pos2();
                            let p2 =
                                (center.to_vec2() + self.pan_offset + pos2.to_vec2()).to_pos2();
                            // Highlight edges touching the selected node.
                            let highlighted = matches!(
                                self.selected_node,
                                Some(sel) if sel == *id || sel == input_id
                            );
                            let stroke = if highlighted {
                                egui::Stroke::new(2.5, egui::Color32::from_rgb(255, 220, 100))
                            } else {
                                egui::Stroke::new(1.5, egui::Color32::GRAY)
                            };
                            painter.line_segment([p1, p2], stroke);
                        }
                    }
                }
            }

            // Second pass: nodes (with click handling).
            let mut new_selection: Option<NodeId> = None;
            for (id, node) in &asg.nodes {
                if let Some(pos) = self.node_positions.get(id) {
                    let node_rect = egui::Rect::from_center_size(
                        (center.to_vec2() + pos.to_vec2() + self.pan_offset).to_pos2(),
                        egui::vec2(NODE_WIDTH, NODE_HEIGHT),
                    );

                    // Skip nodes outside the canvas — minor culling.
                    if !canvas_rect.intersects(node_rect) {
                        continue;
                    }

                    let rounding = egui::Rounding::from(5.0);
                    let fill_color = node_fill_color(node, &asg);

                    let is_selected = self.selected_node == Some(*id);
                    let stroke = if is_selected {
                        egui::Stroke::new(3.0, egui::Color32::from_rgb(255, 220, 100))
                    } else {
                        egui::Stroke::new(1.5, egui::Color32::WHITE)
                    };

                    painter.add(Shape::Rect(eframe::epaint::RectShape {
                        rect: node_rect,
                        rounding,
                        fill: fill_color,
                        stroke,
                        blur_width: 0.0,
                        fill_texture_id: Default::default(),
                        uv: egui::Rect::NOTHING,
                    }));

                    let label = format_node_label(node);
                    painter.text(
                        node_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        label,
                        egui::FontId::proportional(13.0),
                        egui::Color32::BLACK,
                    );

                    // Click interaction over the node rectangle.
                    let node_id_for_click = ui.id().with(("node", *id));
                    let resp = ui.interact(node_rect, node_id_for_click, egui::Sense::click());
                    if resp.clicked() {
                        new_selection = Some(*id);
                    }
                }
            }
            if let Some(sel) = new_selection {
                self.selected_node = Some(sel);
            }
        });
    }
}

// ============================================================
// Educational node descriptions
// ============================================================

/// Parameter-name → translation-key mapping. Inspects the suffix of a
/// parameter name (e.g. `transformer.norm1.beta` → `role_layernorm_beta`)
/// and returns the matching `tr()` key, or `None` when we can't recognise
/// the suffix.
///
/// The naming conventions come from the layer constructors (see `src/nn`):
/// - `LayerNorm` / `BatchNorm` → `.gamma`, `.beta`
/// - `Linear`                  → `.weights`, `.bias`
/// - `Conv2d` / `ConvTranspose2d` → `.weight`, `.bias`
/// - `Embedding`               → `_weight`
/// - `MultiHeadAttention`      → `.w_q`, `.w_k`, `.w_v`, `.w_o`
fn parameter_role_key(name: &str) -> Option<&'static str> {
    let lower = name.to_lowercase();

    // Multi-head attention sub-projections.
    if lower.ends_with(".w_q") {
        return Some("role_attn_query");
    }
    if lower.ends_with(".w_k") {
        return Some("role_attn_key");
    }
    if lower.ends_with(".w_v") {
        return Some("role_attn_value");
    }
    if lower.ends_with(".w_o") {
        return Some("role_attn_output");
    }

    // Norm parameters: try to disambiguate LayerNorm vs BatchNorm by path hint.
    let is_batchnorm_hint = lower.contains("bn") || lower.contains("batch");
    let is_layernorm_hint =
        lower.contains("ln") || lower.contains("layernorm") || lower.contains("norm");

    if lower.ends_with(".gamma") {
        return Some(if is_batchnorm_hint && !is_layernorm_hint {
            "role_batchnorm_gamma"
        } else {
            "role_layernorm_gamma"
        });
    }
    if lower.ends_with(".beta") {
        return Some(if is_batchnorm_hint && !is_layernorm_hint {
            "role_batchnorm_beta"
        } else {
            "role_layernorm_beta"
        });
    }

    // Linear / Conv / Embedding.
    if lower.ends_with(".weights") {
        return Some("role_linear_weights");
    }
    if lower.ends_with("_weight") {
        return Some("role_embedding_weight");
    }
    if lower.ends_with(".weight") {
        return Some("role_conv_weight");
    }
    if lower.ends_with(".bias") {
        return Some("role_bias");
    }

    None
}

/// Builds a rich, language-aware description for a single node.
fn describe_node(node: &Node, asg: &Asg, lang: Lang) -> NodeDescription {
    use NodeType::*;

    let (headline, what, formula, why) = match &node.node_type {
        // ----- Data sources -----
        Input { name } => (
            match lang {
                Lang::En => format!("Input '{}'", name),
                Lang::Ru => format!("Вход '{}'", name),
            },
            match lang {
                Lang::En => "External tensor fed into the graph at runtime. \
                            Has no inputs of its own — it's where data enters."
                    .to_string(),
                Lang::Ru => "Внешний тензор, который подаётся в граф во время исполнения. \
                            У него нет родительских узлов — здесь данные «входят» в граф."
                    .to_string(),
            },
            None,
            match lang {
                Lang::En => "Defines the public API of the graph. Every forward pass binds an \
                            actual tensor to this name in the runtime data map."
                    .to_string(),
                Lang::Ru => "Определяет публичный API графа. Каждый прямой проход привязывает \
                            к этому имени реальный тензор в runtime-словаре."
                    .to_string(),
            },
        ),
        Parameter { name } => (
            match lang {
                Lang::En => format!("Parameter '{}'", name),
                Lang::Ru => format!("Параметр '{}'", name),
            },
            match lang {
                Lang::En => "A trainable weight. Its shape and initial values come from the \
                            `ParameterRegistry` (see the layer constructor)."
                    .to_string(),
                Lang::Ru => "Обучаемый вес. Форма и начальные значения берутся из \
                            `ParameterRegistry` (см. конструктор слоя)."
                    .to_string(),
            },
            None,
            match lang {
                Lang::En => "Parameters are what the optimiser updates. After every backward \
                            pass the gradient w.r.t. this node is fed to SGD/Adam/...".to_string(),
                Lang::Ru => "Параметры — это то, что обновляет оптимизатор. После каждого \
                            обратного прохода соответствующий градиент передаётся в SGD/Adam/...".to_string(),
            },
        ),
        Literal(_) => (
            match lang {
                Lang::En => "Literal constant".to_string(),
                Lang::Ru => "Константа".to_string(),
            },
            match lang {
                Lang::En => "Constant tensor embedded directly into the graph — its value is \
                            baked in at build time and doesn't change between epochs."
                    .to_string(),
                Lang::Ru => "Константный тензор, зашитый прямо в граф — значение фиксировано \
                            на этапе построения и не меняется между эпохами."
                    .to_string(),
            },
            None,
            match lang {
                Lang::En => "Useful for fixed scaling factors (e.g. 1/√d_k in attention) and for \
                            broadcasting scalars in autograd-generated graphs."
                    .to_string(),
                Lang::Ru => "Применяется для фиксированных множителей (например, 1/√d_k в \
                            attention) и для broadcasting скаляров в графе градиентов."
                    .to_string(),
            },
        ),
        External { name, .. } => (
            match lang {
                Lang::En => format!("External '{}'", name),
                Lang::Ru => format!("Внешняя ссылка '{}'", name),
            },
            match lang {
                Lang::En => "Reference to a node living in another ASG (typically the forward \
                            graph). The runtime resolves it through a shared memo cache."
                    .to_string(),
                Lang::Ru => "Ссылка на узел из другого ASG (обычно — из forward-графа). \
                            Runtime находит значение через общий memo-кэш."
                    .to_string(),
            },
            None,
            match lang {
                Lang::En => "The autograd engine generates these so the gradient graph can read \
                            intermediate forward values without re-computing them."
                    .to_string(),
                Lang::Ru => "Autograd создаёт такие узлы, чтобы граф градиентов мог использовать \
                            промежуточные значения forward-прохода без повторного счёта."
                    .to_string(),
            },
        ),

        // ----- Arithmetic -----
        Add(_, _) => (
            "Add".to_string(),
            match lang {
                Lang::En => "Element-wise addition with NumPy-style broadcasting.".to_string(),
                Lang::Ru => "Поэлементное сложение с broadcasting в стиле NumPy.".to_string(),
            },
            Some("y = a + b".to_string()),
            match lang {
                Lang::En => "The workhorse of residual connections (`x + sublayer(x)`) and bias \
                            application after a `MatMul`."
                    .to_string(),
                Lang::Ru => "Основа residual-соединений (`x + sublayer(x)`) и применения bias \
                            после `MatMul`."
                    .to_string(),
            },
        ),
        Subtract(_, _) => (
            "Subtract".to_string(),
            match lang {
                Lang::En => "Element-wise subtraction `a - b`, broadcasting like NumPy.".to_string(),
                Lang::Ru => "Поэлементное вычитание `a - b` с broadcasting'ом.".to_string(),
            },
            Some("y = a - b".to_string()),
            match lang {
                Lang::En => "Used inside MSE loss (`y_pred - y_true`) and centring operations \
                            (`x - mean(x)` in normalisation layers)."
                    .to_string(),
                Lang::Ru => "Используется в MSE-loss (`y_pred - y_true`) и в центрировании \
                            (`x - mean(x)` в слоях нормализации)."
                    .to_string(),
            },
        ),
        Multiply(_, _) => (
            "Multiply".to_string(),
            match lang {
                Lang::En => "Element-wise (Hadamard) product with broadcasting.".to_string(),
                Lang::Ru => "Поэлементное (адамарово) произведение с broadcasting'ом."
                    .to_string(),
            },
            Some("y = a ⊙ b".to_string()),
            match lang {
                Lang::En => "Critical for gating (e.g. SiLU = x · σ(x)), Dropout masks, and \
                            attention weighting (softmax · V)."
                    .to_string(),
                Lang::Ru => "Используется в gating (например, SiLU = x · σ(x)), Dropout-масках \
                            и весах attention (softmax · V)."
                    .to_string(),
            },
        ),
        Divide(_, _) => (
            "Divide".to_string(),
            match lang {
                Lang::En => "Element-wise division `a / b` with broadcasting.".to_string(),
                Lang::Ru => "Поэлементное деление `a / b` с broadcasting'ом.".to_string(),
            },
            Some("y = a / b".to_string()),
            match lang {
                Lang::En => "Appears in normalisation (`x / std`) and any time a quantity has to \
                            be rescaled by a learned or computed factor."
                    .to_string(),
                Lang::Ru => "Встречается в нормализации (`x / std`) и везде, где нужно \
                            масштабировать на обучаемый или вычисленный множитель."
                    .to_string(),
            },
        ),
        MatrixMultiply(_, _) => (
            "MatMul".to_string(),
            match lang {
                Lang::En => "Batched matrix multiplication. Both operands' last two axes are \
                            treated as a matrix; everything before is a batch dimension."
                    .to_string(),
                Lang::Ru => "Батчевое матричное умножение. Последние две оси операндов — \
                            матрица, всё что слева — batch."
                    .to_string(),
            },
            Some("Y = A · B   //  [..., M, K] × [..., K, N] → [..., M, N]".to_string()),
            match lang {
                Lang::En => "The single most expensive op in deep nets — every Linear/Dense \
                            layer and every attention head is a MatMul."
                    .to_string(),
                Lang::Ru => "Самая дорогая операция в нейросетях: каждый Linear-слой и каждая \
                            голова attention — это MatMul."
                    .to_string(),
            },
        ),
        Power(_, _) => (
            "Power".to_string(),
            match lang {
                Lang::En => "Element-wise exponentiation `a ^ b`. The exponent is usually a \
                            broadcast scalar Literal."
                    .to_string(),
                Lang::Ru => "Поэлементное возведение в степень `a ^ b`. Показатель обычно — \
                            скалярная константа с broadcasting'ом."
                    .to_string(),
            },
            Some("y = aᵇ".to_string()),
            match lang {
                Lang::En => "Used inside MSE loss as `(y_pred - y_true)² = pow(diff, 2)` and \
                            anywhere a polynomial term is needed."
                    .to_string(),
                Lang::Ru => "Используется в MSE как `(y_pred - y_true)² = pow(diff, 2)` и \
                            везде, где нужен полиномиальный член."
                    .to_string(),
            },
        ),
        GreaterThan(_, _) => (
            "GreaterThan".to_string(),
            match lang {
                Lang::En => "Element-wise comparison: returns 1.0 where `a > b`, else 0.0."
                    .to_string(),
                Lang::Ru => "Поэлементное сравнение: 1.0 там где `a > b`, иначе 0.0.".to_string(),
            },
            Some("y = (a > b) ? 1.0 : 0.0".to_string()),
            match lang {
                Lang::En => "Building block for masks (e.g. ReLU's gradient is `(x > 0)`) and \
                            for thresholding."
                    .to_string(),
                Lang::Ru => "Используется для построения масок (например, градиент ReLU — \
                            это `(x > 0)`) и для пороговой активации."
                    .to_string(),
            },
        ),

        // ----- Activations -----
        ReLU(_) => (
            "ReLU".to_string(),
            match lang {
                Lang::En => "Rectified Linear Unit. Sets negative values to zero, keeps positive \
                            values unchanged."
                    .to_string(),
                Lang::Ru => "Rectified Linear Unit. Зануляет отрицательные значения, \
                            положительные оставляет как есть."
                    .to_string(),
            },
            Some("y = max(0, x)".to_string()),
            match lang {
                Lang::En => "The default activation in modern CNNs and MLPs — extremely cheap, \
                            doesn't saturate for x > 0, and trains well empirically."
                    .to_string(),
                Lang::Ru => "Стандартная активация в современных CNN и MLP — крайне дешёвая, \
                            не насыщается при x > 0, эмпирически хорошо обучается."
                    .to_string(),
            },
        ),
        Sigmoid(_) => (
            "Sigmoid".to_string(),
            match lang {
                Lang::En => "Squashes any real number into the (0, 1) range.".to_string(),
                Lang::Ru => "Сжимает любое вещественное число в диапазон (0, 1).".to_string(),
            },
            Some("σ(x) = 1 / (1 + e⁻ˣ)".to_string()),
            match lang {
                Lang::En => "Standard final activation for binary classification and any place \
                            where the network needs a probability-like output."
                    .to_string(),
                Lang::Ru => "Стандартная финальная активация для бинарной классификации и \
                            везде, где нужен «вероятностный» выход."
                    .to_string(),
            },
        ),
        Tanh(_) => (
            "Tanh".to_string(),
            match lang {
                Lang::En => "Hyperbolic tangent. Squashes the input into (-1, 1), zero-centred."
                    .to_string(),
                Lang::Ru => "Гиперболический тангенс. Сжимает вход в (-1, 1), симметрично нуля."
                    .to_string(),
            },
            Some("tanh(x) = (eˣ − e⁻ˣ) / (eˣ + e⁻ˣ)".to_string()),
            match lang {
                Lang::En => "Used in classical RNNs (LSTM/GRU gates) and as a smooth, \
                            zero-centred alternative to sigmoid."
                    .to_string(),
                Lang::Ru => "Используется в классических RNN (вентили LSTM/GRU) и как гладкая \
                            симметричная нулю альтернатива сигмоиде."
                    .to_string(),
            },
        ),
        GELU(_) => (
            "GELU".to_string(),
            match lang {
                Lang::En => "Gaussian Error Linear Unit. A smooth, probabilistic relative of ReLU."
                    .to_string(),
                Lang::Ru => "Gaussian Error Linear Unit. Гладкий вероятностный родственник ReLU."
                    .to_string(),
            },
            Some("GELU(x) = x · Φ(x)   //  Φ — CDF стандартной нормали".to_string()),
            match lang {
                Lang::En => "The default FFN activation in modern transformers (BERT, GPT-2, \
                            ViT) — outperforms ReLU on most language benchmarks."
                    .to_string(),
                Lang::Ru => "Активация FFN по умолчанию в современных трансформерах (BERT, \
                            GPT-2, ViT) — обгоняет ReLU на большинстве языковых задач."
                    .to_string(),
            },
        ),
        SiLU(_) => (
            "SiLU (Swish)".to_string(),
            match lang {
                Lang::En => "Sigmoid Linear Unit, also called Swish. A self-gated activation: \
                            multiplies the input by its own sigmoid."
                    .to_string(),
                Lang::Ru => "Sigmoid Linear Unit, она же Swish. Самогейтящаяся активация — \
                            умножает вход на собственную сигмоиду."
                    .to_string(),
            },
            Some("SiLU(x) = x · σ(x)".to_string()),
            match lang {
                Lang::En => "Used in EfficientNet, LLaMA's FFN (combined as SwiGLU), and many \
                            modern image models."
                    .to_string(),
                Lang::Ru => "Применяется в EfficientNet, FFN LLaMA (в сочетании SwiGLU) и \
                            многих современных моделях для изображений."
                    .to_string(),
            },
        ),
        LeakyReLU(_, slope) => (
            "LeakyReLU".to_string(),
            match lang {
                Lang::En => format!(
                    "Like ReLU but lets a small fraction (slope = {}) of the negative input through.",
                    slope
                ),
                Lang::Ru => format!(
                    "Как ReLU, но пропускает небольшую долю (slope = {}) отрицательного входа.",
                    slope
                ),
            },
            Some(format!(
                "y = x if x ≥ 0 else {} · x",
                slope
            )),
            match lang {
                Lang::En => "Used to fix the \"dying ReLU\" problem — neurons whose negative \
                            input would otherwise produce a permanent zero gradient."
                    .to_string(),
                Lang::Ru => "Решает проблему «мёртвого ReLU» — нейронов, у которых \
                            отрицательный вход иначе давал бы нулевой градиент навсегда."
                    .to_string(),
            },
        ),
        ELU(_, alpha) => (
            "ELU".to_string(),
            match lang {
                Lang::En => format!(
                    "Exponential Linear Unit (alpha = {}). Smooth saturating curve for negative \
                     inputs.",
                    alpha
                ),
                Lang::Ru => format!(
                    "Exponential Linear Unit (alpha = {}). Гладкая насыщающаяся кривая для \
                     отрицательного входа.",
                    alpha
                ),
            },
            Some(format!(
                "y = x if x ≥ 0 else {} · (eˣ − 1)",
                alpha
            )),
            match lang {
                Lang::En => "Pushes mean activations closer to zero, which can speed up \
                            convergence vs. ReLU on some tasks."
                    .to_string(),
                Lang::Ru => "Сдвигает среднее активаций к нулю, что иногда ускоряет \
                            сходимость по сравнению с ReLU."
                    .to_string(),
            },
        ),
        Softplus(_, beta) => (
            "Softplus".to_string(),
            match lang {
                Lang::En => format!(
                    "Smooth approximation of ReLU (beta = {}). Differentiable everywhere.",
                    beta
                ),
                Lang::Ru => format!(
                    "Гладкое приближение ReLU (beta = {}). Дифференцируемо везде.",
                    beta
                ),
            },
            Some(format!(
                "y = (1 / {}) · log(1 + e^({} · x))",
                beta, beta
            )),
            match lang {
                Lang::En => "Useful when you need a strictly positive, smooth output (e.g. \
                            standard deviation in a Gaussian likelihood head)."
                    .to_string(),
                Lang::Ru => "Полезно, когда нужен строго положительный гладкий выход \
                            (например, стандартное отклонение в гауссовской голове)."
                    .to_string(),
            },
        ),
        Softmax(_) => (
            "Softmax".to_string(),
            match lang {
                Lang::En => "Turns a vector of arbitrary scores into a probability distribution \
                            (positive values that sum to 1) along the last axis."
                    .to_string(),
                Lang::Ru => "Превращает вектор произвольных оценок в распределение вероятностей \
                            (положительные числа с суммой 1) по последней оси."
                    .to_string(),
            },
            Some("softmax(x)ᵢ = eˣⁱ / Σⱼ eˣʲ".to_string()),
            match lang {
                Lang::En => "Final layer of every classifier, and the heart of attention: \
                            `softmax(Q · Kᵀ / √d_k)` is the attention weight matrix."
                    .to_string(),
                Lang::Ru => "Финальный слой любого классификатора и сердце attention: \
                            `softmax(Q · Kᵀ / √d_k)` — матрица весов внимания."
                    .to_string(),
            },
        ),
        Clamp(_, lo, hi) => (
            "Clamp".to_string(),
            match lang {
                Lang::En => format!(
                    "Clips each element to the range [{}, {}].",
                    lo, hi
                ),
                Lang::Ru => format!("Обрезает каждый элемент в диапазон [{}, {}].", lo, hi),
            },
            Some(format!("y = min({}, max({}, x))", hi, lo)),
            match lang {
                Lang::En => "Useful for gradient clipping per-element, mitigating outliers, and \
                            defining hard activations like ReLU6."
                    .to_string(),
                Lang::Ru => "Используется для поэлементного клиппинга градиентов, борьбы с \
                            выбросами и активаций типа ReLU6."
                    .to_string(),
            },
        ),
        Abs(_) => (
            "Abs".to_string(),
            match lang {
                Lang::En => "Element-wise absolute value.".to_string(),
                Lang::Ru => "Поэлементный модуль.".to_string(),
            },
            Some("y = |x|".to_string()),
            match lang {
                Lang::En => "Heart of L1 loss / L1 regularisation, and used in metric learning \
                            (Manhattan distance)."
                    .to_string(),
                Lang::Ru => "Основа L1-loss / L1-регуляризации и метрик обучения \
                            (манхэттенское расстояние)."
                    .to_string(),
            },
        ),
        Neg(_) => (
            "Neg".to_string(),
            match lang {
                Lang::En => "Element-wise sign flip.".to_string(),
                Lang::Ru => "Поэлементная смена знака.".to_string(),
            },
            Some("y = −x".to_string()),
            match lang {
                Lang::En => "Frequently emitted by autograd — most subtractions decompose into \
                            `Add(a, Neg(b))` in the gradient graph."
                    .to_string(),
                Lang::Ru => "Часто появляется в графе градиентов — вычитания обычно \
                            раскладываются в `Add(a, Neg(b))`."
                    .to_string(),
            },
        ),
        Exp(_) => (
            "Exp".to_string(),
            match lang {
                Lang::En => "Element-wise natural exponent.".to_string(),
                Lang::Ru => "Поэлементная натуральная экспонента.".to_string(),
            },
            Some("y = eˣ".to_string()),
            match lang {
                Lang::En => "Used inside softmax, the negative log-likelihood, and any time a \
                            quantity has to be made strictly positive."
                    .to_string(),
                Lang::Ru => "Применяется в softmax, в отрицательном log-likelihood и везде, \
                            где нужно сделать величину строго положительной."
                    .to_string(),
            },
        ),
        Log(_) => (
            "Log".to_string(),
            match lang {
                Lang::En => "Element-wise natural logarithm.".to_string(),
                Lang::Ru => "Поэлементный натуральный логарифм.".to_string(),
            },
            Some("y = ln(x)".to_string()),
            match lang {
                Lang::En => "Heart of cross-entropy loss: `−Σ y · log(p)`. Also used to convert \
                            multiplicative chains into additive ones for numerical stability."
                    .to_string(),
                Lang::Ru => "Основа кросс-энтропии: `−Σ y · log(p)`. Также используется для \
                            превращения произведений в суммы — для численной устойчивости."
                    .to_string(),
            },
        ),
        Sqrt(_) => (
            "Sqrt".to_string(),
            match lang {
                Lang::En => "Element-wise square root.".to_string(),
                Lang::Ru => "Поэлементный квадратный корень.".to_string(),
            },
            Some("y = √x".to_string()),
            match lang {
                Lang::En => "Used inside LayerNorm/BatchNorm denominator (`√(var + ε)`), in RMS \
                            losses, and in attention's `1/√d_k` scaling."
                    .to_string(),
                Lang::Ru => "Используется в знаменателе LayerNorm/BatchNorm (`√(var + ε)`), \
                            в RMS-loss и в нормировке `1/√d_k` в attention."
                    .to_string(),
            },
        ),

        // ----- Reductions -----
        Sum(_) => (
            "Sum".to_string(),
            match lang {
                Lang::En => "Sums every element of the tensor into a scalar.".to_string(),
                Lang::Ru => "Суммирует все элементы тензора в скаляр.".to_string(),
            },
            Some("y = Σᵢ xᵢ".to_string()),
            match lang {
                Lang::En => "Final reduction of most loss functions — gradient w.r.t. inputs is \
                            then computed via the autograd chain rule from this single number."
                    .to_string(),
                Lang::Ru => "Финальная редукция большинства loss-функций — градиент по входам \
                            берётся autograd'ом цепным правилом от этого скаляра."
                    .to_string(),
            },
        ),
        Mean(_) => (
            "Mean".to_string(),
            match lang {
                Lang::En => "Mean along the last axis (output shape = input shape with the last \
                            axis dropped)."
                    .to_string(),
                Lang::Ru => "Среднее по последней оси (форма выхода = форма входа без \
                            последней оси)."
                    .to_string(),
            },
            Some("μ = (1/N) · Σᵢ xᵢ".to_string()),
            match lang {
                Lang::En => "Centring step of LayerNorm and the mean component of MSE loss."
                    .to_string(),
                Lang::Ru => "Шаг центрирования в LayerNorm и среднее в MSE-loss.".to_string(),
            },
        ),
        Variance(_) => (
            "Variance".to_string(),
            match lang {
                Lang::En => "Variance along the last axis (population formula, divides by N)."
                    .to_string(),
                Lang::Ru => "Дисперсия по последней оси (несмещённая, делится на N).".to_string(),
            },
            Some("σ² = (1/N) · Σᵢ (xᵢ − μ)²".to_string()),
            match lang {
                Lang::En => "Used together with `Mean` to whiten activations in LayerNorm."
                    .to_string(),
                Lang::Ru => "Используется вместе с `Mean` для «обеляющего» преобразования \
                            активаций в LayerNorm."
                    .to_string(),
            },
        ),
        MeanAxis { axis, keepdims, .. } => (
            "MeanAxis".to_string(),
            match lang {
                Lang::En => format!("Mean along axis {} (keepdims = {}).", axis, keepdims),
                Lang::Ru => format!("Среднее по оси {} (keepdims = {}).", axis, keepdims),
            },
            Some("μ = (1/N_axis) · Σᵢ xᵢ".to_string()),
            match lang {
                Lang::En => "Like `Mean`, but reduces along an arbitrary axis. Used in \
                            BatchNorm-style channel statistics."
                    .to_string(),
                Lang::Ru => "Как `Mean`, но редуцирует по произвольной оси. Применяется в \
                            BatchNorm-подобных статистиках по каналу."
                    .to_string(),
            },
        ),
        VarianceAxis { axis, keepdims, .. } => (
            "VarianceAxis".to_string(),
            match lang {
                Lang::En => format!("Variance along axis {} (keepdims = {}).", axis, keepdims),
                Lang::Ru => format!("Дисперсия по оси {} (keepdims = {}).", axis, keepdims),
            },
            Some("σ² = (1/N_axis) · Σᵢ (xᵢ − μ)²".to_string()),
            match lang {
                Lang::En => "Used in BatchNorm-style channel statistics in tandem with \
                            `MeanAxis`."
                    .to_string(),
                Lang::Ru => "Используется в BatchNorm-подобных статистиках по каналу \
                            совместно с `MeanAxis`."
                    .to_string(),
            },
        ),

        // ----- Normalisation -----
        LayerNorm { eps, .. } => (
            "LayerNorm".to_string(),
            match lang {
                Lang::En => format!(
                    "Layer Normalisation: per-sample whitening across the last axis (eps = {}).",
                    eps
                ),
                Lang::Ru => format!(
                    "Layer Normalisation: «обеление» по последней оси, отдельно для каждого \
                     примера (eps = {}).",
                    eps
                ),
            },
            Some("y = γ · (x − μ) / √(σ² + ε) + β".to_string()),
            match lang {
                Lang::En => "The normaliser of choice for transformers. Stabilises training by \
                            keeping per-token activations zero-mean / unit-variance regardless \
                            of batch size."
                    .to_string(),
                Lang::Ru => "Стандартная нормализация в трансформерах. Стабилизирует обучение, \
                            удерживая активации каждого токена в zero-mean / unit-variance \
                            независимо от размера батча."
                    .to_string(),
            },
        ),
        BatchNorm {
            eps, channel_axis, ..
        } => (
            "BatchNorm".to_string(),
            match lang {
                Lang::En => format!(
                    "Batch Normalisation along channel axis {} (eps = {}). Statistics are \
                     computed across batch + spatial dims, separately per channel.",
                    channel_axis, eps
                ),
                Lang::Ru => format!(
                    "Batch Normalisation по канальной оси {} (eps = {}). Статистики считаются \
                     по batch и spatial осям, отдельно для каждого канала.",
                    channel_axis, eps
                ),
            },
            Some("y[..,c,..] = γ[c] · (x[..,c,..] − μ[c]) / √(σ²[c] + ε) + β[c]".to_string()),
            match lang {
                Lang::En => "The normaliser that powered most pre-2020 CNNs. Speeds up \
                            convergence and acts as a mild regulariser thanks to batch noise."
                    .to_string(),
                Lang::Ru => "Нормализация, на которой держались почти все CNN до 2020. \
                            Ускоряет сходимость и работает как мягкий регуляризатор \
                            благодаря шуму батча."
                    .to_string(),
            },
        ),
        LayerNormBackward { .. } => (
            "LayerNormBackward (∂L/∂x)".to_string(),
            match lang {
                Lang::En => "Closed-form gradient of LayerNorm w.r.t. its input. Uses gamma, \
                            the original input, and the upstream gradient."
                    .to_string(),
                Lang::Ru => "Аналитический градиент LayerNorm по входу. Использует gamma, \
                            исходный вход и градиент сверху."
                    .to_string(),
            },
            None,
            match lang {
                Lang::En => "Hand-rolled because naive autograd through `Mean`/`Variance` would \
                            produce a much larger graph; this fused form is faster and more \
                            numerically stable."
                    .to_string(),
                Lang::Ru => "Реализован вручную: наивный autograd через `Mean`/`Variance` дал \
                            бы гораздо больший граф; эта слитная форма быстрее и численно \
                            устойчивее."
                    .to_string(),
            },
        ),
        LayerNormGradGamma { .. } => (
            "LayerNormGradGamma (∂L/∂γ)".to_string(),
            match lang {
                Lang::En => "Gradient of LayerNorm w.r.t. the scale parameter γ.".to_string(),
                Lang::Ru => "Градиент LayerNorm по параметру масштаба γ.".to_string(),
            },
            Some("∂L/∂γ = Σ (∂L/∂y · x_normalised)".to_string()),
            match lang {
                Lang::En => "Feeds straight into the optimiser as the update direction for the \
                            corresponding `γ` parameter."
                    .to_string(),
                Lang::Ru => "Идёт прямо в оптимизатор как направление обновления для \
                            соответствующего параметра `γ`."
                    .to_string(),
            },
        ),
        LayerNormGradBeta { .. } => (
            "LayerNormGradBeta (∂L/∂β)".to_string(),
            match lang {
                Lang::En => "Gradient of LayerNorm w.r.t. the shift parameter β. Trivially the \
                            sum of the upstream gradient over the batch axis."
                    .to_string(),
                Lang::Ru => "Градиент LayerNorm по параметру сдвига β. Просто сумма \
                            входящего градиента по batch-оси."
                    .to_string(),
            },
            Some("∂L/∂β = Σ ∂L/∂y".to_string()),
            match lang {
                Lang::En => "Feeds the optimiser update for `β`.".to_string(),
                Lang::Ru => "Подаётся в оптимизатор для обновления `β`.".to_string(),
            },
        ),
        BatchNormBackward { channel_axis, .. } => (
            "BatchNormBackward (∂L/∂x)".to_string(),
            match lang {
                Lang::En => format!(
                    "Closed-form gradient of BatchNorm w.r.t. its input along channel axis {}.",
                    channel_axis
                ),
                Lang::Ru => format!(
                    "Аналитический градиент BatchNorm по входу вдоль канальной оси {}.",
                    channel_axis
                ),
            },
            None,
            match lang {
                Lang::En => "Same motivation as `LayerNormBackward` — a fused custom op is far \
                            cheaper than autodiff through `MeanAxis`/`VarianceAxis`."
                    .to_string(),
                Lang::Ru => "Та же причина, что и для `LayerNormBackward` — слитная \
                            пользовательская операция гораздо дешевле, чем autograd через \
                            `MeanAxis`/`VarianceAxis`."
                    .to_string(),
            },
        ),
        BatchNormGradGamma { channel_axis, .. } => (
            "BatchNormGradGamma (∂L/∂γ)".to_string(),
            match lang {
                Lang::En => format!(
                    "Per-channel gradient of BatchNorm w.r.t. γ along channel axis {}.",
                    channel_axis
                ),
                Lang::Ru => format!(
                    "Поканальный градиент BatchNorm по γ вдоль канальной оси {}.",
                    channel_axis
                ),
            },
            Some("∂L/∂γ[c] = Σ_{batch+spatial} (∂L/∂y · x_normalised)".to_string()),
            match lang {
                Lang::En => "Drives the per-channel scale update.".to_string(),
                Lang::Ru => "Двигает обновление поканального масштаба.".to_string(),
            },
        ),
        BatchNormGradBeta { channel_axis, .. } => (
            "BatchNormGradBeta (∂L/∂β)".to_string(),
            match lang {
                Lang::En => format!(
                    "Per-channel gradient of BatchNorm w.r.t. β along channel axis {}.",
                    channel_axis
                ),
                Lang::Ru => format!(
                    "Поканальный градиент BatchNorm по β вдоль канальной оси {}.",
                    channel_axis
                ),
            },
            Some("∂L/∂β[c] = Σ_{batch+spatial} ∂L/∂y".to_string()),
            match lang {
                Lang::En => "Drives the per-channel shift update.".to_string(),
                Lang::Ru => "Двигает обновление поканального сдвига.".to_string(),
            },
        ),

        // ----- Convolutions / pooling -----
        Conv2d {
            stride,
            padding,
            dilation,
            groups,
            ..
        } => (
            "Conv2d".to_string(),
            match lang {
                Lang::En => format!(
                    "2D Convolution. stride = {:?}, padding = {:?}, dilation = {:?}, \
                     groups = {}.",
                    stride, padding, dilation, groups
                ),
                Lang::Ru => format!(
                    "Двумерная свёртка. stride = {:?}, padding = {:?}, dilation = {:?}, \
                     groups = {}.",
                    stride, padding, dilation, groups
                ),
            },
            Some(
                "y[n,c_out,h,w] = Σ_{c_in,kh,kw} x[n, c_in, h·s+kh·d, w·s+kw·d] · W[c_out,c_in,kh,kw]"
                    .to_string(),
            ),
            match lang {
                Lang::En => "The fundamental block of CNNs. Local receptive fields + weight \
                            sharing make it the natural fit for images and other grid-structured \
                            data."
                    .to_string(),
                Lang::Ru => "Базовый блок CNN. Локальные рецептивные поля + общие веса делают \
                            его естественным для изображений и других сеточно-структурированных \
                            данных."
                    .to_string(),
            },
        ),
        ConvTranspose2d {
            stride,
            padding,
            output_padding,
            dilation,
            groups,
            ..
        } => (
            "ConvTranspose2d".to_string(),
            match lang {
                Lang::En => format!(
                    "Transposed 2D convolution. stride = {:?}, padding = {:?}, \
                     output_padding = {:?}, dilation = {:?}, groups = {}.",
                    stride, padding, output_padding, dilation, groups
                ),
                Lang::Ru => format!(
                    "Транспонированная двумерная свёртка. stride = {:?}, padding = {:?}, \
                     output_padding = {:?}, dilation = {:?}, groups = {}.",
                    stride, padding, output_padding, dilation, groups
                ),
            },
            None,
            match lang {
                Lang::En => "Up-samples spatial resolution. The bread-and-butter op of decoders \
                            in U-Net, autoencoders, and image-generation models."
                    .to_string(),
                Lang::Ru => "Увеличивает пространственное разрешение. Базовая операция \
                            декодеров U-Net, автоэнкодеров и генеративных моделей."
                    .to_string(),
            },
        ),
        Conv2dBackwardInput { .. } => (
            "Conv2dBackwardInput (∂L/∂x)".to_string(),
            match lang {
                Lang::En => "Gradient of Conv2d w.r.t. its input. Implemented as a transposed \
                            convolution of the upstream gradient with the kernel weights."
                    .to_string(),
                Lang::Ru => "Градиент Conv2d по входу. Реализуется как транспонированная \
                            свёртка градиента сверху с ядром."
                    .to_string(),
            },
            None,
            match lang {
                Lang::En => "Custom op so the framework doesn't have to recompute the input \
                            shape and stride bookkeeping at autograd time."
                    .to_string(),
                Lang::Ru => "Спец-операция: фреймворку не нужно во время autograd заново \
                            пересчитывать форму входа и stride."
                    .to_string(),
            },
        ),
        Conv2dBackwardWeight { .. } => (
            "Conv2dBackwardWeight (∂L/∂W)".to_string(),
            match lang {
                Lang::En => "Gradient of Conv2d w.r.t. the kernel weights. Implemented as a \
                            convolution of the input with the upstream gradient."
                    .to_string(),
                Lang::Ru => "Градиент Conv2d по ядру. Реализуется как свёртка входа с \
                            градиентом сверху."
                    .to_string(),
            },
            None,
            match lang {
                Lang::En => "Drives the optimiser update for the convolution kernel.".to_string(),
                Lang::Ru => "Двигает обновление свёрточного ядра в оптимизаторе.".to_string(),
            },
        ),
        MaxPool2d {
            kernel_size,
            stride,
            ..
        } => (
            "MaxPool2d".to_string(),
            match lang {
                Lang::En => format!(
                    "Max pooling 2D, kernel = {:?}, stride = {:?}. Takes the maximum value \
                     inside each window.",
                    kernel_size, stride
                ),
                Lang::Ru => format!(
                    "Двумерный max pooling, kernel = {:?}, stride = {:?}. Берёт максимум в \
                     каждом окне.",
                    kernel_size, stride
                ),
            },
            None,
            match lang {
                Lang::En => "Down-samples spatial dimensions while preserving the strongest \
                            features. Provides translation invariance."
                    .to_string(),
                Lang::Ru => "Уменьшает пространственные размеры, сохраняя сильнейшие признаки. \
                            Обеспечивает инвариантность к сдвигам."
                    .to_string(),
            },
        ),
        AvgPool2d {
            kernel_size,
            stride,
            padding,
            ..
        } => (
            "AvgPool2d".to_string(),
            match lang {
                Lang::En => format!(
                    "Average pooling 2D, kernel = {:?}, stride = {:?}, padding = {:?}.",
                    kernel_size, stride, padding
                ),
                Lang::Ru => format!(
                    "Двумерный average pooling, kernel = {:?}, stride = {:?}, padding = {:?}.",
                    kernel_size, stride, padding
                ),
            },
            None,
            match lang {
                Lang::En => "Smoother alternative to MaxPool — used in classical Inception/ResNet \
                            and global average pooling heads."
                    .to_string(),
                Lang::Ru => "Более «гладкая» альтернатива MaxPool — используется в Inception/\
                            ResNet и в global average pooling-головах."
                    .to_string(),
            },
        ),
        AdaptiveAvgPool2d { output_size, .. } => (
            "AdaptiveAvgPool2d".to_string(),
            match lang {
                Lang::En => format!(
                    "Adaptive average pooling: pools to fixed output size {:?} regardless of \
                     input resolution.",
                    output_size
                ),
                Lang::Ru => format!(
                    "Adaptive average pooling: приводит к фиксированному выходу {:?} независимо \
                     от разрешения входа.",
                    output_size
                ),
            },
            None,
            match lang {
                Lang::En => "Lets a CNN classifier accept arbitrary input sizes — the head sees \
                            a fixed-shape feature map either way."
                    .to_string(),
                Lang::Ru => "Позволяет CNN-классификатору принимать произвольные размеры входа \
                            — голова всегда видит фичи фиксированной формы."
                    .to_string(),
            },
        ),
        MaxUnpool2d { .. } => (
            "MaxUnpool2d".to_string(),
            match lang {
                Lang::En => "Backward pass of MaxPool2d: routes the gradient back only to the \
                            element that was the maximum in each pooling window."
                    .to_string(),
                Lang::Ru => "Обратный проход MaxPool2d: пускает градиент только в тот элемент, \
                            который был максимумом в окне пулинга."
                    .to_string(),
            },
            None,
            match lang {
                Lang::En => "Generated automatically by autograd when MaxPool2d is part of a \
                            differentiable graph."
                    .to_string(),
                Lang::Ru => "Создаётся autograd'ом, когда MaxPool2d — часть дифференцируемого \
                            графа."
                    .to_string(),
            },
        ),
        AvgUnpool2d { .. } => (
            "AvgUnpool2d".to_string(),
            match lang {
                Lang::En => "Backward pass of AvgPool2d: distributes the gradient uniformly back \
                            across the pooling window."
                    .to_string(),
                Lang::Ru => "Обратный проход AvgPool2d: равномерно распределяет градиент по окну \
                            пулинга."
                    .to_string(),
            },
            None,
            match lang {
                Lang::En => "Generated automatically by autograd when AvgPool2d is part of a \
                            differentiable graph."
                    .to_string(),
                Lang::Ru => "Создаётся autograd'ом, когда AvgPool2d — часть дифференцируемого \
                            графа."
                    .to_string(),
            },
        ),

        // ----- Embedding -----
        Embedding { .. } => (
            "Embedding".to_string(),
            match lang {
                Lang::En => "Looks up dense vectors by integer indices into the embedding table."
                    .to_string(),
                Lang::Ru => "Ищет плотные векторы по целочисленным индексам в таблице \
                            эмбеддингов."
                    .to_string(),
            },
            Some("y[i, :] = W[indices[i], :]".to_string()),
            match lang {
                Lang::En => "First layer of every NLP model — turns token IDs into the \
                            continuous vectors the rest of the network operates on."
                    .to_string(),
                Lang::Ru => "Первый слой любой NLP-модели — превращает ID токенов в \
                            непрерывные векторы, с которыми работает остальная сеть."
                    .to_string(),
            },
        ),
        EmbeddingGrad { num_embeddings, .. } => (
            "EmbeddingGrad".to_string(),
            match lang {
                Lang::En => format!(
                    "Backward pass of Embedding: scatter-add of upstream gradients into a \
                     [{}, embedding_dim] table by index.",
                    num_embeddings
                ),
                Lang::Ru => format!(
                    "Обратный проход Embedding: scatter-add градиентов сверху в таблицу \
                     [{}, embedding_dim] по индексам.",
                    num_embeddings
                ),
            },
            None,
            match lang {
                Lang::En => "Each token's embedding only sees gradients from positions where \
                            the token actually appeared — that's why it's a scatter, not a \
                            dense MatMul."
                    .to_string(),
                Lang::Ru => "Эмбеддинг каждого токена получает градиенты только из позиций, \
                            где он действительно встретился — поэтому это scatter, а не \
                            плотный MatMul."
                    .to_string(),
            },
        ),

        // ----- Shape ops -----
        Reshape(_, _) => (
            "Reshape".to_string(),
            match lang {
                Lang::En => "Reinterprets the same data with a different shape, without copying."
                    .to_string(),
                Lang::Ru => "Переинтерпретирует те же данные с другой формой, без копирования."
                    .to_string(),
            },
            None,
            match lang {
                Lang::En => "Used to flatten before a Linear head, or to split an MHA tensor \
                            into per-head pieces."
                    .to_string(),
                Lang::Ru => "Используется для flatten перед Linear-головой и для разбиения \
                            тензора MHA по головам."
                    .to_string(),
            },
        ),
        Transpose(_, a, b) => (
            "Transpose".to_string(),
            match lang {
                Lang::En => format!("Swaps axes {} and {}.", a, b),
                Lang::Ru => format!("Меняет местами оси {} и {}.", a, b),
            },
            None,
            match lang {
                Lang::En => "Critical for attention's `Q · Kᵀ`, for switching between channels-\
                            first and channels-last layouts, and for general tensor algebra."
                    .to_string(),
                Lang::Ru => "Критично для `Q · Kᵀ` в attention, для переключения между \
                            channels-first / channels-last и для тензорной алгебры в целом."
                    .to_string(),
            },
        ),
        Broadcast(_, _) => (
            "Broadcast".to_string(),
            match lang {
                Lang::En => "Expands a smaller tensor to match the shape of a larger one without \
                            copying (the runtime simulates the expansion on the fly)."
                    .to_string(),
                Lang::Ru => "Раздувает меньший тензор до формы большего без копирования \
                            (runtime эмулирует расширение на лету)."
                    .to_string(),
            },
            None,
            match lang {
                Lang::En => "Emitted by autograd whenever a scalar gradient (e.g. from `Sum`) \
                            needs to be propagated back to a tensor input."
                    .to_string(),
                Lang::Ru => "Создаётся autograd'ом, когда скалярный градиент (например, от \
                            `Sum`) нужно распространить обратно на тензорный вход."
                    .to_string(),
            },
        ),
        ReduceSumTo(_, _) => (
            "ReduceSumTo".to_string(),
            match lang {
                Lang::En => "Sums the source tensor along the axes that were broadcast, until \
                            its shape matches the target."
                    .to_string(),
                Lang::Ru => "Суммирует исходный тензор по тем осям, по которым был broadcast, \
                            до формы целевого тензора."
                    .to_string(),
            },
            None,
            match lang {
                Lang::En => "The autograd dual of `Broadcast` — it's how a broadcast operation's \
                            gradient is contracted back to the original input shape."
                    .to_string(),
                Lang::Ru => "Двойственная операция к `Broadcast` в autograd — именно так \
                            градиент broadcast-операции сжимается до исходной формы."
                    .to_string(),
            },
        ),
        Slice {
            axis, start, end, ..
        } => (
            "Slice".to_string(),
            match lang {
                Lang::En => format!(
                    "Takes the sub-tensor `[..., {}..{}, ...]` along axis {}.",
                    start, end, axis
                ),
                Lang::Ru => format!(
                    "Берёт под-тензор `[..., {}..{}, ...]` по оси {}.",
                    start, end, axis
                ),
            },
            None,
            match lang {
                Lang::En => "Used to split a packed Q/K/V projection into separate tensors and \
                            for many other index-based extractions."
                    .to_string(),
                Lang::Ru => "Используется для разделения упакованной Q/K/V проекции и многих \
                            других извлечений по индексу."
                    .to_string(),
            },
        ),
        Concat { axis, inputs } => (
            "Concat".to_string(),
            match lang {
                Lang::En => format!(
                    "Concatenates {} tensors along axis {}.",
                    inputs.len(),
                    axis
                ),
                Lang::Ru => format!(
                    "Конкатенирует {} тензоров по оси {}.",
                    inputs.len(),
                    axis
                ),
            },
            None,
            match lang {
                Lang::En => "The natural inverse of Slice — used to merge per-head MHA outputs \
                            and to combine feature maps in U-Net skip connections."
                    .to_string(),
                Lang::Ru => "Естественная обратная к Slice — объединяет выходы голов MHA и \
                            склеивает feature maps в skip-соединениях U-Net."
                    .to_string(),
            },
        ),
        SliceBackward {
            axis,
            start,
            full_size,
            ..
        } => (
            "SliceBackward".to_string(),
            match lang {
                Lang::En => format!(
                    "Zero-pads the upstream gradient back to size {} along axis {}, placing it \
                     at offset {}. The gradient dual of `Slice`.",
                    full_size, axis, start
                ),
                Lang::Ru => format!(
                    "Дополняет нулями градиент сверху до размера {} по оси {} с офсетом {}. \
                     Двойственная операция к `Slice`.",
                    full_size, axis, start
                ),
            },
            None,
            match lang {
                Lang::En => "Generated by autograd whenever Slice is part of a differentiable \
                            graph."
                    .to_string(),
                Lang::Ru => "Создаётся autograd'ом всякий раз, когда Slice — часть \
                            дифференцируемого графа."
                    .to_string(),
            },
        ),

        DropoutMask { p, .. } => (
            "DropoutMask".to_string(),
            match lang {
                Lang::En => format!(
                    "Bernoulli mask for Dropout: each element is 1/(1−p) with probability {}, \
                     else 0.",
                    1.0 - p
                ),
                Lang::Ru => format!(
                    "Bernoulli-маска для Dropout: каждый элемент равен 1/(1−p) с вероятностью \
                     {}, иначе 0.",
                    1.0 - p
                ),
            },
            Some("dropout(x) = x · DropoutMask(x, p)".to_string()),
            match lang {
                Lang::En => "Cached in the forward memo so the backward pass sees the same \
                            mask. Acts as a strong regulariser by randomly zeroing activations \
                            during training."
                    .to_string(),
                Lang::Ru => "Кэшируется в forward memo, чтобы обратный проход видел ту же \
                            маску. Сильный регуляризатор: случайно зануляет активации во \
                            время обучения."
                    .to_string(),
            },
        ),

        // ----- Control flow / I/O (rare in normal training graphs) -----
        If { .. } => (
            "If".to_string(),
            match lang {
                Lang::En => "Conditional execution — runs one of two sub-graphs depending on a \
                            scalar condition."
                    .to_string(),
                Lang::Ru => "Условное исполнение — запускает один из двух подграфов в \
                            зависимости от скалярного условия."
                    .to_string(),
            },
            None,
            match lang {
                Lang::En => "Currently used only for advanced graph constructions; standard \
                            training graphs do not produce this node."
                    .to_string(),
                Lang::Ru => "Пока используется только в продвинутых конструкциях графа; \
                            обычные training-графы такой узел не создают."
                    .to_string(),
            },
        ),
        ForLoop { .. } => (
            "ForLoop".to_string(),
            match lang {
                Lang::En => "Loop construct that re-runs a sub-graph over an iterable.".to_string(),
                Lang::Ru => "Цикл, повторяющий подграф по итерируемому объекту.".to_string(),
            },
            None,
            match lang {
                Lang::En => "Reserved for future RNN-style use; not produced by current layers."
                    .to_string(),
                Lang::Ru => "Зарезервировано под будущее использование (RNN); текущие слои не \
                            создают такой узел."
                    .to_string(),
            },
        ),
        FunctionDefinition { name, .. } => (
            format!("FunctionDefinition '{}'", name),
            match lang {
                Lang::En => "Defines a reusable sub-graph that can be invoked via `FunctionCall`."
                    .to_string(),
                Lang::Ru => "Определяет переиспользуемый подграф, вызываемый через \
                            `FunctionCall`."
                    .to_string(),
            },
            None,
            match lang {
                Lang::En => "Forward-looking primitive for graph composition.".to_string(),
                Lang::Ru => "Заготовка для будущей композиции графов.".to_string(),
            },
        ),
        FunctionCall { .. } => (
            "FunctionCall".to_string(),
            match lang {
                Lang::En => "Calls a previously defined `FunctionDefinition` with the provided \
                            arguments."
                    .to_string(),
                Lang::Ru => "Вызывает определённый ранее `FunctionDefinition` с переданными \
                            аргументами."
                    .to_string(),
            },
            None,
            match lang {
                Lang::En => "Forward-looking primitive for graph composition.".to_string(),
                Lang::Ru => "Заготовка для будущей композиции графов.".to_string(),
            },
        ),
        Print(_) => (
            "Print".to_string(),
            match lang {
                Lang::En => "Prints the input tensor to stdout during execution. Side-effect \
                            only — its output value is the input untouched."
                    .to_string(),
                Lang::Ru => "Печатает входной тензор в stdout во время исполнения. Только \
                            побочный эффект — выход равен входу."
                    .to_string(),
            },
            None,
            match lang {
                Lang::En => "Useful as a debug probe inside a graph.".to_string(),
                Lang::Ru => "Полезно как отладочный «зонд» внутри графа.".to_string(),
            },
        ),
    };

    // Build optional context: for parameters, recognise their role from the name;
    // for graph outputs, mark them as such.
    let mut context: Option<String> = None;

    if let NodeType::Parameter { name } = &node.node_type {
        if let Some(role_key) = parameter_role_key(name) {
            context = Some(tr(lang, role_key).to_string());
        }
    }

    if asg.outputs.contains(&node.id) {
        let prefix = match lang {
            Lang::En => "This is a graph output — the value of the whole forward pass terminates \
                        at this node."
                .to_string(),
            Lang::Ru => "Это выход графа — значение всего forward-прохода заканчивается на этом \
                        узле."
                .to_string(),
        };
        context = Some(match context {
            Some(existing) => format!("{}\n\n{}", prefix, existing),
            None => prefix,
        });
    }

    NodeDescription {
        headline,
        what,
        formula,
        why,
        context,
    }
}

// ============================================================
// Node rendering helpers
// ============================================================

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
        other => human_node_type(other),
    };

    format!("ID: {}\n{}{}", node.id, type_str, shape_info)
}

/// Compact human-readable name for a `NodeType` (no debug-spam params).
fn human_node_type(t: &NodeType) -> String {
    match t {
        NodeType::Input { name } => format!("Input({})", name),
        NodeType::Parameter { name } => format!("Parameter({})", name),
        NodeType::Literal(_) => "Literal".into(),
        NodeType::External { name, .. } => format!("External({})", name),
        NodeType::Add(_, _) => "Add".into(),
        NodeType::Subtract(_, _) => "Subtract".into(),
        NodeType::Multiply(_, _) => "Multiply".into(),
        NodeType::Divide(_, _) => "Divide".into(),
        NodeType::MatrixMultiply(_, _) => "MatMul".into(),
        NodeType::GreaterThan(_, _) => "GreaterThan".into(),
        NodeType::Power(_, _) => "Power".into(),
        NodeType::ReLU(_) => "ReLU".into(),
        NodeType::Sigmoid(_) => "Sigmoid".into(),
        NodeType::Tanh(_) => "Tanh".into(),
        NodeType::GELU(_) => "GELU".into(),
        NodeType::SiLU(_) => "SiLU".into(),
        NodeType::LeakyReLU(_, _) => "LeakyReLU".into(),
        NodeType::ELU(_, _) => "ELU".into(),
        NodeType::Softplus(_, _) => "Softplus".into(),
        NodeType::Softmax(_) => "Softmax".into(),
        NodeType::Clamp(_, _, _) => "Clamp".into(),
        NodeType::Abs(_) => "Abs".into(),
        NodeType::Neg(_) => "Neg".into(),
        NodeType::Exp(_) => "Exp".into(),
        NodeType::Log(_) => "Log".into(),
        NodeType::Sqrt(_) => "Sqrt".into(),
        NodeType::Sum(_) => "Sum".into(),
        NodeType::Mean(_) => "Mean".into(),
        NodeType::Variance(_) => "Variance".into(),
        NodeType::MeanAxis { .. } => "MeanAxis".into(),
        NodeType::VarianceAxis { .. } => "VarianceAxis".into(),
        NodeType::Reshape(_, _) => "Reshape".into(),
        NodeType::Transpose(_, _, _) => "Transpose".into(),
        NodeType::Broadcast(_, _) => "Broadcast".into(),
        NodeType::ReduceSumTo(_, _) => "ReduceSumTo".into(),
        NodeType::Slice { .. } => "Slice".into(),
        NodeType::Concat { .. } => "Concat".into(),
        NodeType::SliceBackward { .. } => "SliceBackward".into(),
        NodeType::DropoutMask { .. } => "DropoutMask".into(),
        NodeType::Conv2d { .. } => "Conv2d".into(),
        NodeType::ConvTranspose2d { .. } => "ConvTranspose2d".into(),
        NodeType::Conv2dBackwardInput { .. } => "Conv2dBackwardInput".into(),
        NodeType::Conv2dBackwardWeight { .. } => "Conv2dBackwardWeight".into(),
        NodeType::MaxPool2d { .. } => "MaxPool2d".into(),
        NodeType::MaxUnpool2d { .. } => "MaxUnpool2d".into(),
        NodeType::AvgPool2d { .. } => "AvgPool2d".into(),
        NodeType::AvgUnpool2d { .. } => "AvgUnpool2d".into(),
        NodeType::AdaptiveAvgPool2d { .. } => "AdaptiveAvgPool2d".into(),
        NodeType::Embedding { .. } => "Embedding".into(),
        NodeType::EmbeddingGrad { .. } => "EmbeddingGrad".into(),
        NodeType::LayerNorm { .. } => "LayerNorm".into(),
        NodeType::LayerNormBackward { .. } => "LayerNormBackward".into(),
        NodeType::LayerNormGradGamma { .. } => "LayerNormGradGamma".into(),
        NodeType::LayerNormGradBeta { .. } => "LayerNormGradBeta".into(),
        NodeType::BatchNorm { .. } => "BatchNorm".into(),
        NodeType::BatchNormBackward { .. } => "BatchNormBackward".into(),
        NodeType::BatchNormGradGamma { .. } => "BatchNormGradGamma".into(),
        NodeType::BatchNormGradBeta { .. } => "BatchNormGradBeta".into(),
        NodeType::If { .. } => "If".into(),
        NodeType::ForLoop { .. } => "ForLoop".into(),
        NodeType::FunctionDefinition { .. } => "FunctionDefinition".into(),
        NodeType::FunctionCall { .. } => "FunctionCall".into(),
        NodeType::Print(_) => "Print".into(),
    }
}

/// Returns the fill colour for a node based on its category.
fn node_fill_color(node: &Node, asg: &Asg) -> egui::Color32 {
    if asg.outputs.contains(&node.id) {
        return egui::Color32::from_rgb(255, 200, 130); // brighter peach for output
    }
    match &node.node_type {
        NodeType::Input { .. } => egui::Color32::from_rgb(180, 230, 200), // green
        NodeType::Parameter { .. } => egui::Color32::from_rgb(150, 210, 230), // teal
        NodeType::External { .. } => egui::Color32::from_rgb(230, 230, 250), // lavender
        NodeType::Literal(_) => egui::Color32::from_rgb(220, 220, 220),   // gray

        NodeType::Add(..)
        | NodeType::Subtract(..)
        | NodeType::Multiply(..)
        | NodeType::Divide(..)
        | NodeType::MatrixMultiply(..)
        | NodeType::Power(..)
        | NodeType::GreaterThan(..)
        | NodeType::Neg(..)
        | NodeType::Abs(..)
        | NodeType::Exp(..)
        | NodeType::Log(..)
        | NodeType::Sqrt(..) => egui::Color32::from_rgb(208, 225, 255), // light blue

        NodeType::ReLU(..)
        | NodeType::Sigmoid(..)
        | NodeType::Tanh(..)
        | NodeType::GELU(..)
        | NodeType::SiLU(..)
        | NodeType::LeakyReLU(..)
        | NodeType::ELU(..)
        | NodeType::Softplus(..)
        | NodeType::Softmax(..)
        | NodeType::Clamp(..) => egui::Color32::from_rgb(255, 240, 180), // pale yellow

        NodeType::Sum(..)
        | NodeType::Mean(..)
        | NodeType::Variance(..)
        | NodeType::MeanAxis { .. }
        | NodeType::VarianceAxis { .. } => egui::Color32::from_rgb(255, 215, 180), // orange

        NodeType::LayerNorm { .. } | NodeType::BatchNorm { .. } => {
            egui::Color32::from_rgb(255, 200, 220)
        } // pink

        NodeType::Conv2d { .. }
        | NodeType::ConvTranspose2d { .. }
        | NodeType::MaxPool2d { .. }
        | NodeType::AvgPool2d { .. }
        | NodeType::AdaptiveAvgPool2d { .. } => egui::Color32::from_rgb(220, 200, 250), // purple

        NodeType::Embedding { .. } => egui::Color32::from_rgb(200, 220, 250), // dim blue

        NodeType::Reshape(..)
        | NodeType::Transpose(..)
        | NodeType::Broadcast(..)
        | NodeType::Slice { .. }
        | NodeType::Concat { .. } => egui::Color32::from_rgb(230, 230, 200), // pale olive

        NodeType::DropoutMask { .. } => egui::Color32::from_rgb(250, 230, 210),

        // Gradient ops — slightly muted versions of forward colours.
        NodeType::Conv2dBackwardInput { .. }
        | NodeType::Conv2dBackwardWeight { .. }
        | NodeType::LayerNormBackward { .. }
        | NodeType::LayerNormGradGamma { .. }
        | NodeType::LayerNormGradBeta { .. }
        | NodeType::BatchNormBackward { .. }
        | NodeType::BatchNormGradGamma { .. }
        | NodeType::BatchNormGradBeta { .. }
        | NodeType::EmbeddingGrad { .. }
        | NodeType::MaxUnpool2d { .. }
        | NodeType::AvgUnpool2d { .. }
        | NodeType::SliceBackward { .. }
        | NodeType::ReduceSumTo(..) => egui::Color32::from_rgb(190, 190, 200),

        _ => egui::Color32::WHITE,
    }
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
        | NodeType::Exp(a)
        | NodeType::Abs(a)
        | NodeType::Neg(a)
        | NodeType::Tanh(a)
        | NodeType::GELU(a)
        | NodeType::SiLU(a)
        | NodeType::Transpose(a, _, _) => vec![*a],

        NodeType::LeakyReLU(a, _)
        | NodeType::ELU(a, _)
        | NodeType::Softplus(a, _)
        | NodeType::Clamp(a, _, _) => vec![*a],

        NodeType::Slice { input, .. } => vec![*input],
        NodeType::SliceBackward { grad_output, .. } => vec![*grad_output],
        NodeType::Concat { inputs, .. } => inputs.clone(),
        NodeType::DropoutMask { shape_provider, .. } => vec![*shape_provider],

        NodeType::MeanAxis { input, .. } | NodeType::VarianceAxis { input, .. } => vec![*input],

        NodeType::MaxPool2d { input, .. } | NodeType::AvgPool2d { input, .. } => vec![*input],
        NodeType::AdaptiveAvgPool2d { input, .. } => vec![*input],
        NodeType::MaxUnpool2d {
            input,
            original_input,
            ..
        } => vec![*input, *original_input],
        NodeType::AvgUnpool2d {
            input,
            original_input,
            ..
        } => vec![*input, *original_input],

        NodeType::Conv2d {
            input,
            weight,
            bias,
            ..
        } => {
            let mut deps = vec![*input, *weight];
            if let Some(b) = bias {
                deps.push(*b);
            }
            deps
        }
        NodeType::ConvTranspose2d {
            input,
            weight,
            bias,
            ..
        } => {
            let mut deps = vec![*input, *weight];
            if let Some(b) = bias {
                deps.push(*b);
            }
            deps
        }
        NodeType::Conv2dBackwardInput {
            grad_output,
            weight,
            ..
        } => vec![*grad_output, *weight],
        NodeType::Conv2dBackwardWeight {
            grad_output, input, ..
        } => vec![*grad_output, *input],

        NodeType::Embedding { indices, weight } => vec![*indices, *weight],
        NodeType::EmbeddingGrad {
            grad_output,
            indices,
            ..
        } => vec![*grad_output, *indices],

        NodeType::LayerNorm {
            input, gamma, beta, ..
        } => vec![*input, *gamma, *beta],
        NodeType::LayerNormBackward {
            grad_output,
            input,
            gamma,
            ..
        } => vec![*grad_output, *input, *gamma],
        NodeType::LayerNormGradGamma {
            grad_output, input, ..
        } => vec![*grad_output, *input],
        NodeType::LayerNormGradBeta { grad_output } => vec![*grad_output],

        NodeType::BatchNorm {
            input, gamma, beta, ..
        } => vec![*input, *gamma, *beta],
        NodeType::BatchNormBackward {
            grad_output,
            input,
            gamma,
            ..
        } => vec![*grad_output, *input, *gamma],
        NodeType::BatchNormGradGamma {
            grad_output, input, ..
        } => vec![*grad_output, *input],
        NodeType::BatchNormGradBeta { grad_output, .. } => vec![*grad_output],

        _ => vec![],
    }
}
