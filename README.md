![RustyASG Logo](logo.png)

# RustyASG — a graph-based deep learning engine in Rust

**RustyASG** is a modern, experimental deep learning framework written in pure
Rust with a unique feature: **live, interactive graph visualization**. Its
defining idea is an architecture built around an **Abstract Semantic Graph
(ASG)** — a symbolic representation of your computation that can be analysed,
differentiated, optimised and executed on multiple backends.

Unlike eager-execution frameworks (PyTorch, TensorFlow 2.x), RustyASG first
*builds* a full computation graph, then runs static analysis, autograd, and
finally hands the result to a CPU or GPU backend. The GPU backend uses
[`wgpu`](https://wgpu.rs), so the same code runs on Vulkan, Metal, DX12 and
WebGPU.

> Russian version of this document: [README.ru.md](README.ru.md).

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
- **Built-in interactive graph visualiser.** A native `egui` window renders
  your graph in real time. Pure Rust, zero external dependencies — no
  Graphviz, no web stack.
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
  full 141-test suite.
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
│    Abstract Semantic Graph (ASG)  │◀──┐ (sent to the GUI)
│       (symbolic computation)      │   │
└─────────────────┬─────────────────┘   │
                  │                     │
        ┌─────────┼─────────┐           │
        ▼         ▼         ▼           │
  ┌─────────┐ ┌───────┐ ┌────────┐      │
  │Autograd │ │Runtime│ │ Graph  │──────┘
  │(graph → │ │       │ │Viewer  │
  │ graph)  │ │       │ │(egui)  │
  └─────────┘ └───┬───┘ └────────┘
                  │
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

# Same demo with the live egui graph visualiser
cargo run --release -- --visualize

# Run one of the standalone examples
cargo run --release --example xor                    # 2-layer MLP, solves XOR
cargo run --release --example linear_regression      # y = wx + b
cargo run --release --example pattern_recognition    # 4-class MLP, 100% accuracy
cargo run --release --example mnist                  # MLP on synthetic MNIST, 100%
cargo run --release --example cnn_classifier         # Conv2d + Pool + Linear, 100%
cargo run --release --example transformer_classifier # attention-style classifier
```

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
cargo test --release                                         # full suite (141 tests)
cargo test --release --lib                                   # unit tests only (87)
cargo test --release --test grad_check                       # numerical grad check (8)
cargo test --release --test gpu_backend -- --test-threads=1  # GPU↔CPU parity (46)
```

**141 tests — all green:**
- 87 library unit tests (activations, autograd, optimizers, data, metrics…)
- 8 numerical gradient checks for LayerNorm and Conv2d backward rules
- 46 GPU↔CPU parity tests — every GPU operation compared against the CPU
  reference at `1e-5` tolerance

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full plan. In short:

- **v0.1 – v0.2** — ASG core, autograd, layer zoo, optimizers, SafeTensors,
  wgpu backend for base operations.
- **v0.3** — declarative layer API (`ParameterRegistry`), complete GPU
  coverage (LayerNorm, Conv2d backward, pooling, embedding,
  ConvTranspose2d, Slice/Concat), full RoPE, CI, clippy-clean, thin-LTO,
  CNN classifier example.
- **v0.5 (planned)** — kernel fusion, GPU buffer pool, mixed precision
  (f16), inference-only mode, criterion benchmarks, tiny GPT / ViT
  starters.
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
