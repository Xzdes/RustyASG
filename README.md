![RustyASG Logo](logo.png)

# RustyASG — a graph-based deep learning engine in Rust

**RustyASG** is a modern, experimental deep learning framework written in pure
Rust with a unique feature: a **live, interactive graph visualiser with an
educational Node Inspector** (v0.4.1+). Click any node in your model and
the side panel will explain — in plain English or Russian — *what* the
operation does, the *formula* behind it, *why* it appears in real
architectures, and (for trainable parameters) the *role* it plays in
*your* specific model.

Its defining idea is an architecture built around an **Abstract Semantic
Graph (ASG)** — a symbolic representation of your computation that can be
analysed, differentiated, optimised and executed on multiple backends.

Unlike eager-execution frameworks (PyTorch, TensorFlow 2.x), RustyASG first
*builds* a full computation graph, then runs static analysis, autograd, and
finally hands the result to a CPU or GPU backend. The GPU backend uses
[`wgpu`](https://wgpu.rs), so the same code runs on Vulkan, Metal, DX12 and
WebGPU.

> 🌐 **Multilingual UI.** Launch the visualiser with `--lang en` (default)
> or `--lang ru`. Every label, panel header, and educational description
> has a parallel English / Russian translation.
>
> 📄 **Russian version of this document:** [README.ru.md](README.ru.md).

[![Crates.io](https://img.shields.io/crates/v/rustyasg.svg)](https://crates.io/crates/rustyasg)
[![Documentation](https://docs.rs/rustyasg/badge.svg)](https://docs.rs/rustyasg)
[![CI](https://github.com/Xzdes/RustyAsg/actions/workflows/ci.yml/badge.svg)](https://github.com/Xzdes/RustyAsg/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust 1.75+](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## Design principles

- **Performance through graphs.** Define-then-run makes global optimisations
  (kernel fusion, static memory planning) possible in a way eager
  frameworks cannot match.
- **Rust safety.** No UB, no data races, no segfaults — the properties that
  matter most during long training runs.
- **Control and transparency.** The computation graph is inspectable,
  modifiable, and — critically — **visualisable in real time**. Debugging
  and understanding a model become dramatically easier.
- **Educational value.** A clear, readable reference for how modern DL
  frameworks actually work under the hood, from the symbolic tensor API
  all the way down to WGSL shaders and graph-level autograd.

## What's inside

- **Declarative layer API (v0.3+).** `Linear::new(ctx, "fc1", 784, 128)`
  automatically registers the layer's parameter shapes and initialisers
  with the graph context. `GraphContext::init_parameters()` samples the
  weights. No more manual `HashMap<String, Shape>`, no more string-matching
  layer names in user code.
- **Built-in interactive graph visualiser with an educational Node
  Inspector (v0.4.1+).** A native `egui` window renders your graph in
  real time as the model trains. **Click any node** and the side panel
  explains, in plain English or Russian:
  - *what* the operation does (one-paragraph plain-language description),
  - the *formula* in monospace
    (e.g. `softmax(xᵢ) = eˣⁱ / Σⱼ eˣʲ`,
    `y = γ · (x − μ)/√(σ² + ε) + β`),
  - *why* it shows up in real architectures (GELU → "the default FFN
    activation in modern transformers"; MatMul → "the single most
    expensive op in deep nets"; …),
  - for `Parameter` nodes: the *role* in this specific model — the
    inspector parses the parameter name (`norm1.gamma`, `mha.w_q`,
    `fc1.weights`, …) and explains what it does plus how it was
    initialised.

  Plus a **live loss chart** in a docked bottom panel, **edge
  highlighting** for the selected node, **per-category color coding**
  (parameter / input / activation / arithmetic / reduction /
  normalisation / convolution / pooling / shape op / gradient op /
  output), and a **two-language UI** selectable at startup with
  `--lang en|ru`. Pure Rust, zero external dependencies — no
  Graphviz, no web stack. **No other Rust DL framework offers this.**
- **Graph-to-graph autograd.** Gradient computation itself is an ASG — it
  can be analysed, optimised, and visualised the same way as the forward
  graph.
- **Two backends:**
  - ✅ **CPU** — complete reference implementation on `ndarray`.
  - ✅ **GPU (wgpu)** — LayerNorm (fwd + bwd), Conv2d (fwd + bwd), Pooling
    (Max/Avg/Adaptive), Embedding, ConvTranspose2d, Slice/Concat.
    `TransformerBlock` trains end-to-end on GPU. 46 parity tests verify
    every GPU op matches the CPU reference to `1e-5`.
- **Static analysis.** `ShapeInference` validates the graph before
  execution, catching shape errors at graph-build time rather than at
  runtime deep inside a training loop.
- **Transformers and CNNs covered:** Multi-Head Attention with causal and
  padding masks, LayerNorm, FeedForward, Conv2d / ConvTranspose2d, pooling
  layers, positional encodings (Sinusoidal, Learned, **full RoPE**, ALiBi),
  and Slice/Concat primitives with autograd.
- **Training stack:** SGD / Adam / AdamW / RMSprop, five LR schedulers,
  gradient clipping, 14 loss functions, 9 standard weight initialisers
  (Xavier / Kaiming / Normal / …).
- **Data & metrics:** `Dataset` / `DataLoader` with samplers and
  transforms, classification and regression metrics, `EarlyStopping`.
- **Serialization:** SafeTensors plus a rotating checkpoint manager.
- **CI/CD:** GitHub Actions matrix across Linux / Windows / macOS. Strict
  `cargo fmt`, `cargo clippy -- -D warnings`, strict `cargo doc`, and the
  full 150-test suite.
- **crates.io ready:** complete metadata, thin-LTO release profile,
  `docs.rs` configuration, tight published-crate size.

## A 20-line XOR (v0.3+)

```rust
use rustyasg::losses::mse_loss;
use rustyasg::nn::{Linear, Module};
use rustyasg::tensor::{GraphContext, Tensor};
use std::{cell::RefCell, rc::Rc};

let ctx = Rc::new(RefCell::new(GraphContext::new()));

let x      = Tensor::new_input(&ctx, "x");
let y_true = Tensor::new_input(&ctx, "y_true");

// Every layer self-registers its parameter shapes + initialisers
// on the GraphContext. No user-side shape bookkeeping.
let fc1 = Linear::new(&ctx, "fc1", 2, 8);
let fc2 = Linear::new(&ctx, "fc2", 8, 1);

let y_pred = fc2.forward(&fc1.forward(&x).relu()).sigmoid();
let loss   = mse_loss(&y_pred, &y_true);

// Training loop — see examples/xor.rs for the full version.
```

## Architecture

```
┌───────────────────────────────────┐
│     User-facing API (Tensor)      │
└─────────────────┬─────────────────┘
                  │ (builds the graph)
                  ▼
┌───────────────────────────────────┐
│    Abstract Semantic Graph (ASG)  │◀────┐ (GraphReady)
│       (symbolic computation)      │     │
└─────────────────┬─────────────────┘     │
                  │                       │
        ┌─────────┼─────────┐             │
        ▼         ▼         ▼             │
  ┌─────────┐ ┌───────┐ ┌─────────────────────────┐
  │Autograd │ │Runtime│ │   Graph Viewer (egui)   │
  │(graph → │ │       │ │  ┌───────────────────┐  │
  │ graph)  │ │       │ │  │ Canvas (DAG view) │  │
  └─────────┘ └───┬───┘ │  ├───────────────────┤  │
                  │     │  │  Node Inspector   │  │
                  │     │  │ (what / formula / │  │
                  │     │  │  why / role) EN×RU│  │
                  │     │  ├───────────────────┤  │
                  │ ────┼─▶│  Live loss chart  │  │ (EpochDone)
                  │     │  └───────────────────┘  │
                  │     │   --lang en | ru        │
                  │     └─────────────────────────┘
            ┌─────┴─────┐
            ▼           ▼
       ┌────────┐  ┌────────┐
       │  CPU   │  │  GPU   │
       │Backend │  │(wgpu)  │
       └────────┘  └────────┘
```

## Getting started

**Prerequisites:** Rust 1.75+ (`rustup install stable`).

```bash
git clone https://github.com/Xzdes/RustyAsg.git
cd RustyAsg

# Train the bundled TransformerBlock demo (CPU by default)
cargo run --release

# Same demo on the GPU backend via wgpu
cargo run --release -- --gpu

# Same demo with the live egui graph visualiser (English UI by default)
cargo run --release -- --visualize

# Visualiser with Russian UI
cargo run --release -- --visualize --lang ru

# Visualiser + GPU backend at the same time
cargo run --release -- --visualize --gpu --lang en

# Run one of the standalone examples
cargo run --release --example xor                    # 2-layer MLP, solves XOR
cargo run --release --example linear_regression      # y = wx + b
cargo run --release --example pattern_recognition    # 4-class MLP, 100% accuracy
cargo run --release --example mnist                  # MLP on synthetic MNIST, 100%
cargo run --release --example cnn_classifier         # Conv2d + Pool + Linear, 100%
cargo run --release --example transformer_classifier # attention-style classifier
```

### CLI flags reference

| Flag | Default | Description |
|------|---------|-------------|
| `--visualize` (`-v`) | off | Open the interactive `egui` graph viewer |
| `--gpu` | off | Use GPU (`wgpu`) backend instead of CPU |
| `--lang <code>` | `en` | UI language for the visualiser. Accepts `en` / `english` / `ru` / `russian`. Only meaningful with `--visualize` |

## Interactive visualiser

> RustyASG ships with a built-in, native `egui` graph viewer that runs
> on the same thread as the GUI while training happens in a background
> thread. Everything you see updates **in real time** as the model
> trains — no Graphviz, no web stack, no Python.

### Launching it

```bash
# English UI (default)
cargo run --release -- --visualize

# Russian UI
cargo run --release -- --visualize --lang ru

# Either flag form is accepted
cargo run --release -- --visualize --lang english
cargo run --release -- --visualize --lang russian

# Combine with the GPU backend
cargo run --release -- --visualize --gpu --lang en
```

### What you see

The window is split into three live-updating panels:

1. **Central canvas** — the full ASG drawn as a layered DAG. Nodes are
   coloured by category:
   - 🟢 **Green** — `Input`
   - 🟦 **Teal** — `Parameter` (trainable weight)
   - ⬜ **Light gray** — `Literal` (constant)
   - 💜 **Lavender** — `External` (cross-graph reference)
   - 🟦 **Light blue** — arithmetic (`Add`, `Mul`, `MatMul`, …)
   - 💛 **Pale yellow** — activations (`ReLU`, `GELU`, `Softmax`, …)
   - 🟧 **Orange** — reductions (`Sum`, `Mean`, `Variance`, …)
   - 🌸 **Pink** — normalisation (`LayerNorm`, `BatchNorm`)
   - 🟪 **Purple** — convolutions / pooling
   - 🫒 **Pale olive** — shape ops (`Reshape`, `Transpose`, `Slice`, …)
   - 🩶 **Muted gray** — gradient-only nodes (`*Backward`, `*Grad*`)
   - 🍑 **Bright peach** — graph **output**
2. **Right side panel — Node Inspector.** Empty until you click a node,
   then it shows the educational sections:
   - **What this node does** (plain English / Russian)
   - **Formula** (in monospace)
   - **Why it's used**
   - **Role in this model** (for `Parameter` nodes — parses the name
     suffix to identify γ/β, weights, biases, Q/K/V projections, …
     and explains what initialisation was used)
   - **Technical details** (collapsible: id, name, type, shape, dtype,
     graph-output flag, list of input nodes)
3. **Bottom panel — Live loss chart.** XY plot of training loss vs.
   epoch. Auto-rescales as new epochs arrive from the compute thread.

### Controls

| Action | What it does |
|--------|--------------|
| **Click** on a node | Selects it — Inspector populates, the node gets a yellow border, all incident edges turn amber |
| **Click and drag** on empty canvas | Pans the view |
| **Drag** the side-panel border | Resizes the Inspector and loss-chart panels |

### Switching languages

The language is fixed at startup time. Pass `--lang en` (default) or
`--lang ru` when launching. Aliases `english` / `russian` are also
accepted, as are `English` / `Russian` (case-insensitive). An unknown
value falls back to English with a warning printed to stderr —
something like:

```
Unknown --lang value 'fr', falling back to English. Try `en` or `ru`.
```

Adding a third language is a one-row change to the `tr()` lookup table
in `src/gui_viewer.rs` — every UI string lives there and has a parallel
EN/RU pair.

### Background: why this is unique

PyTorch can `print(model)` and TensorBoard can render a static graph
*after* training. RustyASG is the only Rust DL framework where the
graph is a first-class runtime object **and** there's a live native
viewer that explains every node as it executes. Phases B–E of the
[Interactive Model Lab roadmap](ROADMAP.md) will turn this into a full
right-click-to-edit, drag-and-drop model lab — see the roadmap for
details.

## Examples

| File | Architecture | Task | Result |
|------|-------------|------|--------|
| [`xor.rs`](examples/xor.rs) | MLP 2→8→1 | XOR | loss < 0.0001 |
| [`linear_regression.rs`](examples/linear_regression.rs) | y = wx + b | learn y = 2x + 1 | error 0.0001 |
| [`pattern_recognition.rs`](examples/pattern_recognition.rs) | MLP 64→32→16→4 | 4 image patterns | 100% |
| [`mnist.rs`](examples/mnist.rs) | MLP 784→128→64→10 | synthetic MNIST | 100% |
| [`cnn_classifier.rs`](examples/cnn_classifier.rs) | Conv2d + Pool + Linear | 3 classes, 8×8 | 100% |
| [`transformer_classifier.rs`](examples/transformer_classifier.rs) | attention-like MLP | sequence patterns | converges |

## Testing

```bash
cargo test --release                                         # full suite (150 tests)
cargo test --release --lib                                   # unit tests only (93)
cargo test --release --test grad_check                       # numerical grad check (8)
cargo test --release --test gpu_backend -- --test-threads=1  # GPU↔CPU parity (46)
```

**150 tests — all green:**
- 93 library unit tests (activations, autograd, optimizers, data, metrics…)
- 9 numerical gradient checks for LayerNorm, Conv2d, BatchNorm backward rules
- 48 GPU↔CPU parity tests — every GPU operation compared against the CPU
  reference at `1e-5` tolerance

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full plan. In short:

- **v0.1 – v0.2** — ASG core, autograd, layer zoo, optimizers, SafeTensors,
  wgpu backend for base operations.
- **v0.3** — declarative layer API (`ParameterRegistry`), complete GPU
  coverage (LayerNorm, Conv2d backward, pooling, embedding,
  ConvTranspose2d, Slice/Concat), full RoPE, CI, clippy-clean, thin-LTO,
  CNN classifier example.
- **v0.4.0** — real `Dropout` and correct `BatchNorm` (both were silently
  broken in v0.3), native GPU `Concat`, full GPU `Conv2d` with `groups`
  and `dilation`. 150 tests green.
- **v0.4.1 — Phase A of the Interactive Model Lab.** Click a node → side
  panel explains *what* the operation does, the *formula*, *why* it's
  used, and (for parameters) its *role* in the model. Edge highlighting,
  per-category color coding, live loss chart, two-language UI
  (`--lang en|ru`). Library API unchanged.
- **v0.5 (planned)** — kernel fusion, GPU buffer pool, mixed precision
  (f16), inference-only mode, criterion benchmarks, tiny GPT / ViT
  starters, `MultiHeadAttention` refactor for `seq_len > 1`.
- **v0.6+ (planned) — Interactive Model Lab Phases B–E.** Right-click →
  swap activations and watch loss recompute live. Insert Dropout into a
  running training loop with one click. Drag-and-drop model builder.
  **No other Rust DL framework can do this** because none of them have a
  first-class graph plus a live visualiser. Phases B–E in `ROADMAP.md`.
- **v1.0** — production-ready API, published documentation, ONNX export,
  WebAssembly target.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The short version:

1. Fork and branch: `git checkout -b feature/xyz`.
2. `cargo build --release --all-targets`, `cargo test --release`.
3. `cargo fmt --all` and
   `cargo clippy --release --all-targets -- -D warnings`.
4. Add an entry under `[Unreleased]` in `CHANGELOG.md`.
5. Open a pull request.

Bug reports and feature suggestions are welcome via GitHub Issues.

## License

MIT — see [LICENSE](LICENSE).
