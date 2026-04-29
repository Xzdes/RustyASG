# RustyASG Roadmap

This document tracks what is already implemented and where the project is
heading next.

## Current status

RustyASG v0.4.1 ships **Phase A** of the **Interactive Model Lab** — the
live `egui` graph viewer is now an educational diagnostic tool that
explains every node's purpose, formula, and role in the model in plain
English or Russian. The library is clippy-clean under `-D warnings`,
rustdoc builds strictly with
`RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links"`, and the full test
suite (150 tests: 93 lib + 48 GPU + 9 grad check, plus 2 ignored
diagnostic tests) is green on every supported platform in CI.

## Implemented

### Framework core
- [x] ASG (Abstract Semantic Graph) architecture
- [x] Define-then-run execution model
- [x] Symbolic `Tensor` API
- [x] CPU backend (`ndarray`)
- [x] GPU backend (`wgpu` — Vulkan/Metal/DX12/WebGPU)
- [x] Shape inference and static analysis
- [x] Interactive graph visualiser (`egui`)

### Automatic differentiation
- [x] Graph-to-graph autograd (the gradient is itself a separate ASG)
- [x] Arithmetic: Add, Sub, Mul, Div, MatMul, Power
- [x] Activations: ReLU, Sigmoid, Tanh, Softmax, GELU, SiLU, LeakyReLU,
      ELU, Softplus, Abs, Clamp
- [x] Reductions: Sum, Mean, Variance
- [x] Pooling gradients: MaxPool2d (MaxUnpool2d), AvgPool2d (AvgUnpool2d)
- [x] Embedding gradient (EmbeddingGrad, scatter-add)
- [x] LayerNorm backward: specialised `LayerNormBackward`,
      `LayerNormGradGamma`, `LayerNormGradBeta`
- [x] Conv2d backward: `Conv2dBackwardInput` and `Conv2dBackwardWeight`
- [x] Slice / Concat gradients (enables end-to-end RoPE)
- [x] Transpose and reshape

### Neural network layers (`nn`)
- [x] `Linear` — declarative API with `ParameterRegistry`
- [x] `LayerNorm`, `BatchNorm`
- [x] `Dropout`, `SpatialDropout`
- [x] `Conv2d` (stride, padding, dilation, groups)
- [x] `ConvTranspose2d`
- [x] `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d` (global pooling)
- [x] `Embedding` (lookup layer for NLP)
- [x] `MultiHeadAttention` with causal and padding masks
- [x] `TransformerBlock` (Pre-LN residual block)
- [x] `FeedForward`

### Positional encodings
- [x] Sinusoidal positional encoding
- [x] Learned positional embedding
- [x] **Rotary Position Embedding (RoPE)** — full split-half
      implementation via `Slice` + `Concat` (v0.3.1)
- [x] ALiBi

### Weight initialisers (`nn::init`)
- [x] Zeros, Ones, Constant
- [x] Uniform, Normal (with mean/std)
- [x] Xavier uniform/normal
- [x] Kaiming uniform/normal

### Optimisers
- [x] SGD (momentum, weight decay, Nesterov)
- [x] Adam, AdamW
- [x] RMSprop

### Learning-rate schedulers
- [x] StepLR, ExponentialLR
- [x] CosineAnnealingLR
- [x] LinearWarmupLR
- [x] WarmupCosineAnnealingLR

### Gradient clipping
- [x] `clip_grad_norm`, `clip_grad_value`

### Loss functions
- [x] MSE, L1, Smooth-L1 / Huber
- [x] Cross-entropy (with optional label smoothing)
- [x] Binary cross-entropy, BCE-with-logits
- [x] KL divergence, NLL
- [x] Hinge, squared hinge, focal
- [x] Cosine embedding, triplet margin, margin ranking

### Serialization
- [x] SafeTensors save/load
- [x] Checkpoint system (weights + optimiser state + metadata)
- [x] `CheckpointManager` with automatic rotation

### Data pipeline
- [x] `Dataset` trait, `InMemoryDataset`
- [x] `MapDataset` (lazy transforms)
- [x] `ConcatDataset`, `SubsetDataset`
- [x] `train_test_split`
- [x] `DataLoader` with batching
- [x] Samplers: Sequential, Random, Weighted, Batch
- [x] Transforms: Normalize, MinMaxScale, OneHot, Clip, Log, Flatten,
      RandomNoise

### Metrics
- [x] Classification: Accuracy, Precision, Recall, F1-score
- [x] Binary and multi-class confusion matrices
- [x] Top-K accuracy
- [x] Regression: MSE, RMSE, MAE, R², MAPE, explained variance, max error
- [x] Running statistics: `RunningMean`, `RunningStd`, EMA
- [x] `MetricLogger`
- [x] `EarlyStopping`

### CI / release engineering
- [x] GitHub Actions matrix — Linux / Windows / macOS
- [x] Strict `cargo fmt --check`
- [x] Strict `cargo clippy --all-targets -- -D warnings`
- [x] Strict `cargo doc` with `-D rustdoc::broken_intra_doc_links`
- [x] Full `Cargo.toml` metadata (rust-version, homepage, docs URL, …)
- [x] `[package.metadata.docs.rs]` for nice docs.rs builds
- [x] Thin-LTO release profile, `strip = "debuginfo"`
- [x] `exclude = ["logo.png", ...]` — published crate stays small

---

## Comparison with other Rust DL frameworks

| Feature | RustyASG | Burn | Candle | PyTorch |
|---------|----------|------|--------|---------|
| Language | Rust | Rust | Rust | Python/C++ |
| Execution | Define-then-run | Eager | Eager | Both |
| GPU backend | `wgpu` | `wgpu`/CUDA | CUDA | CUDA |
| Visualisation | **Built-in** | No | No | TensorBoard |
| Autograd | **Graph-to-graph** | Tape | Tape | Tape |
| SafeTensors | Yes | Yes | Yes | Yes |
| Transformers | Yes | Yes | Yes | Yes |
| Conv2d | Yes | Yes | Yes | Yes |
| Distributed | Planned | Yes | No | Yes |
| WebAssembly | Planned | Yes | No | No |

### What makes RustyASG unique
1. **Built-in real-time graph visualisation** — nothing comparable in the
   Rust DL space.
2. **Define-then-run** enables global optimisations (kernel fusion,
   memory planning).
3. **Pure Rust** — no Python interpreter, no CUDA-SDK install dance.
4. **Educational** — a readable end-to-end reference for how modern DL
   frameworks work.

---

## Release history

### v0.4.1 — Phase A: Interactive Model Lab (read-only) (April 2026)
- **Educational Node Inspector.** Click any node in the live graph viewer
  → side panel explains *what* the operation does, the *formula*, *why*
  it shows up in real models, and — for parameters — the *role* in this
  specific model (γ/β of LayerNorm, Q/K/V projections, weight matrix
  initialisation, …). Plain English or Russian, selected at startup with
  `--lang en|ru`. Coverage: every `NodeType` produced by current layers,
  including all gradient-only ops.
- **Live loss chart.** Bottom panel auto-renders an XY plot of training
  loss vs. epoch, auto-rescaling on min/max, updated as the compute
  thread emits `EpochDone`.
- **Edge highlighting + per-category color coding.** Selected node's
  incident edges get amber highlight; nodes are filled by category
  (parameter / input / activation / arithmetic / reduction /
  normalisation / convolution / pooling / shape op / gradient op /
  output).
- **Two-language UI.** New `--lang en|ru` CLI flag — every label and
  description has a parallel translation in `tr()`.
- **`ComputeUpdate` channel protocol.** Typed `GraphReady` / `EpochDone`
  enum replaces the raw `Asg` send; future phases will add a reverse
  `GuiCommand` channel for mutations.
- Library API unchanged — purely additive release.

### v0.4.0 — Phase 7: Fix what was broken (April 2026)
- **Real `Dropout`.** Previously a no-op; now a `NodeType::DropoutMask`
  that samples Bernoulli on every forward run, with the mask cached in
  `forward_memo` so backward sees the same values via the standard
  `Multiply` rule.
- **Correct `BatchNorm`.** Previously reduced over the *last* axis (via
  `Tensor::mean`); now a specialised `NodeType::BatchNorm` reduces over
  every axis except `channel_axis`. Forward, backward, `grad_gamma`,
  `grad_beta` are all hand-verified by unit tests.
- **Native GPU `Concat`.** Previously round-tripped through CPU; now a
  multi-dispatch WGSL kernel that copies each input into its slice of
  the output buffer with per-axis offset arithmetic.
- **Full GPU `Conv2d`.** Forward and both backward kernels now support
  `groups > 1` (depthwise / grouped convolutions) and `dilation > 1`
  (dilated convolutions). New parity tests cover both regimes.
- New ASG primitives: `DropoutMask`, `MeanAxis`, `VarianceAxis`,
  `BatchNorm`/`Backward`/`GradGamma`/`GradBeta`. CPU + GPU + autograd.
- 9 new tests; full suite at 150 green.

### v0.3.1 — Pre-release polish (April 2026)
- `cargo fmt` applied across the full tree.
- `cargo clippy --all-targets -- -D warnings` clean everywhere; library
  only allows three deliberate design-motivated lints
  (`too_many_arguments`, `type_complexity`, `should_implement_trait`).
- Strict `cargo doc` passes — 10 previously-broken intra-doc links
  fixed.
- Extended `Cargo.toml` metadata, thin-LTO release profile, `exclude`
  list so the published crate is small (no `logo.png`).
- CI split into four dedicated jobs: fmt / clippy / doc / test-matrix.
- New `cnn_classifier` example: first full Conv2d-based example
  (Conv2d + pool + Linear + Adam), 100% accuracy on a tiny synthetic
  dataset.
- Dual-language README (`README.md` + `README.ru.md`), entire project
  documentation converted to English.

### v0.3.0 — Declarative layer API + GPU completeness (April 2026)

**Phase 2 — API reliability**
- Declarative `ParameterRegistry`. Every `nn::*` layer registers its
  parameter shapes and initialisers with `GraphContext`:
  ```rust
  let fc = Linear::new(&ctx, "fc1", 784, 128);        // shapes auto-registered
  ctx.borrow().init_parameters(&mut runtime_data);    // Xavier / Zeros sampled
  ShapeInference::run_with_context(&mut g, &ctx.borrow(), &inputs)?;
  ```
- New `nn::init` module with 9 standard initialisers.
- `Tensor::new_parameter_with_shape(ctx, name, shape, init)` —
  preferred constructor for trainable weights.
- `GraphContext::{register_parameter_meta, parameter_meta,
  parameter_registry, build_shape_map, init_parameters}`.
- **Breaking**: every `nn::*` layer constructor now takes dimension
  arguments. `main.rs` no longer uses string-matching to infer
  parameter shapes (`name.contains("w_q")` is gone).

**Phase 3 — GPU completeness**
- `LayerNorm` forward plus three backward WGSL shaders
  (`LayerNormBackward`, `LayerNormGradGamma`, `LayerNormGradBeta`).
  `TransformerBlock` trains end-to-end on GPU.
- `Conv2dBackwardInput`, `Conv2dBackwardWeight` (groups=1, dilation=(1,1)).
- `MaxPool2d`, `MaxUnpool2d`, `AvgPool2d`, `AvgUnpool2d`,
  `AdaptiveAvgPool2d`.
- `Embedding`, `EmbeddingGrad`.
- `ConvTranspose2d` forward with bias.
- `dispatch_rowwise` helper for per-row / per-column WGSL kernels.

**Phase 5 — Ecosystem polish**
- `Slice`, `Concat`, `SliceBackward` primitives with full CPU + GPU +
  shape inference + autograd support.
- `Tensor::slice(axis, start, end)` and `Tensor::concat(others, axis)`
  methods.
- **Full RoPE** via `Slice` + `Concat` —
  `[x1, x2] → [x1·cos - x2·sin, x1·sin + x2·cos]`, mathematically
  correct and end-to-end differentiable. Previously a stub that added
  `cos` as bias.
- GitHub Actions CI, `CHANGELOG.md`, `CONTRIBUTING.md`.

### v0.2.0 — Phase 1 cleanup (January 2026)
- `main.rs` refactored to consume the library via `use rustyasg::*`
  (eliminated ~100 false-positive warnings).
- Deprecated `rand::thread_rng` → `rand::rng`.
- All unused imports / variables across the tree fixed.
- Reached 0 warnings in `cargo build --release --all-targets`.

### v0.1.0 — Core (pre-history)
ASG, autograd, CPU + basic GPU ops, initial layer zoo, optimizers,
SafeTensors, interactive visualiser.

---

## Planned for v0.5 — Performance & production

- **Inference-only mode.** Skip autograd overhead when only forward is
  needed.
- **Kernel fusion.** Combine `MatMul + Bias + Activation` into a single
  WGSL kernel; fuse LayerNorm sub-ops.
- **GPU buffer pool.** Reuse allocations between training steps instead
  of allocating fresh each epoch.
- **Mixed precision (f16).** GPU-side f16 with loss-scaling.
- **Better errors.** Replace remaining ~125 `unwrap()` call sites in
  library code with typed `RustyAsgError`.
- **Criterion benchmarks.** Measured comparisons against Burn and
  Candle on representative workloads.
- **Tiny GPT block.** End-to-end GPT-style example. Blocked on a
  `MultiHeadAttention` refactor — current `split_heads` hardcodes
  `batch=1, seq_len=1` via a literal `reshape(vec![1, 1, num_heads,
  head_dim])`. Need dynamic shape support.
- **Vision Transformer starter.** Patch embedding + TransformerBlock
  stack.

## Planned for v0.6+ — Interactive Model Lab

This is the direction that uses RustyASG's **single biggest unique
advantage** — the ASG is a first-class object that can be edited at
runtime, and we already render it live with `egui`. No other Rust DL
framework can do this; PyTorch can't either (TensorBoard is read-only).
The vision: turn RustyASG into a node-based visual ML lab — drop a
graph node, wire it up, see the model retrain on the fly.

Implementation is split into five phases of increasing scope. Each
phase is independently shippable and adds visible value.

### Phase A — Read-only inspection ✅ DONE (v0.4.1, April 2026)
Foundation work: make the visualiser a real diagnostic tool, not just a
structure-renderer.
- ✅ **Click on a node** opens a side panel with type, shape, dtype,
  parameter name, graph-output flag, and a list of inputs.
- ✅ **Educational descriptions** (Phase A++): the side panel explains
  *what* the operation does, the *formula*, *why* it shows up in real
  models, and — for parameters — the *role* in this specific model
  (`mha.w_q` → "Query projection of Multi-Head Attention", `norm1.gamma`
  → "Learnable scale γ of LayerNorm; initialised to ones", etc.).
- ✅ **Edge highlighting** when a node is selected — incident edges are
  drawn in amber so the dataflow is visible at a glance.
- ✅ **Color coding** for node categories: inputs / parameters / literals /
  arithmetic / activations / reductions / normalisation / convolutions /
  pooling / shape ops / gradient ops, with a brighter peach for the
  output node.
- ✅ **Live loss chart** in a docked bottom panel, auto-rescaling per
  min/max as `EpochDone` arrives from the compute thread.
- ✅ **`ComputeUpdate` channel protocol** — typed `GraphReady` /
  `EpochDone` enum replaces the raw `Asg` send.
- ✅ **Two-language UI** (English / Russian) via `--lang en|ru`.
- ⏳ Hover-on-edge tensor stats and live forward-value preview are
  deferred to Phase B (require the planned reverse `GuiCommand` channel
  to request specific tensor values from the compute thread).

### Phase B — Atomic mutations (1–2 sessions)
First real interactivity. Mutations that don't change shape compatibility,
so re-execution is cheap.
- `Asg::replace_node_type(id, new_type)` — same input/output shape
  required.
- **Right-click on activation → "Replace with..."** menu:
  ReLU ↔ GELU ↔ SiLU ↔ Sigmoid ↔ Tanh ↔ LeakyReLU ↔ ELU.
- **Edit Literal** — slider for scalars, editable table for arrays.
- **Pause / Step / Reset weights** controls in the GUI.
- Mutation marks a `dirty_subtree`; only that subtree is re-evaluated,
  not the full graph.
- Loss chart updates immediately after mutation.

The wow-effect demo: open `main.rs` running, swap `ReLU → GELU` in the
TransformerBlock through GUI, watch loss curve change in real time.

### Phase C — Insert / delete operations on edges (2–3 sessions)
Topology changes — graph structure mutates, not just values.
- `Asg::insert_between(parent, child, op)` — insert a node on an edge.
- `Asg::delete_passthrough(id)` — remove a single-input/single-output
  node, splice the edges.
- **Right-click on edge → "Insert here..."** with palette of ops
  (Dropout, BatchNorm, ReLU, LayerNorm, ...).
- **Right-click on node → "Delete"** with cascade or splice options.
- Cascade re-shape inference after structural change.
- Rebuild gradient graph automatically (the gradient graph is
  invalidated whenever the forward graph changes).
- Validation: insertion only allowed when shape compatibility is
  satisfied; user gets a red-highlighted preview when not.

The wow-effect demo: training a network, click on the edge between two
linear layers, insert Dropout(0.3), watch overfitting reduce live.

### Phase D — Parameter editing (3–5 sessions)
Layer dimensions become editable.
- Edit `Linear::out_features` → resize weights (re-init from same
  initialiser) → propagate new shape to downstream layers, fail-fast on
  incompatibility.
- Edit `Conv2d` kernel size, stride, padding, dilation, groups → re-shape-infer
  downstream.
- Edit `MultiHeadAttention` `embed_dim` and `num_heads` (requires Tiny
  GPT MHA refactor first).
- **Layer-level editor panel** with a preview of which downstream
  parameters need reshaping; "Apply" only commits if the cascade
  succeeds.
- Save / load partial training checkpoints — so editing doesn't lose
  the parameters that didn't change shape.

The wow-effect demo: experiment with `hidden_dim` of FeedForward
without restarting Python — it's not Python.

### Phase E — Build-from-scratch (multi-week project)
The full PyTorch-replacement experience without writing any Rust.
- **Drag-and-drop palette** of all operations on the side.
- **Snap-to-port** edge routing — only shape-compatible connections
  light up green.
- **Save / load model architecture as JSON** — share models without code.
- **Pre-built templates**: "Add LeNet block", "Add Transformer block",
  "Add ResNet residual block".
- **Live training panel**: dataset selection, optimiser, loss function
  — all configurable from GUI.
- **Export to Rust code** so the visually-built model can graduate to
  a production training script.

The wow-effect demo: build, train, and inspect a small CNN end-to-end
without touching a `cargo run`.

### Why this is the project's biggest USP

| What other frameworks have | RustyASG with Phases A–E |
|---|---|
| `print(model)` static text | Live, interactive node graph |
| TensorBoard read-only viz | Read + edit + re-execute |
| `model.layers[0] = ...` Python rebuild | One-click GUI mutation, live-train |
| Read-only PyTorch profiler | Inspect + mutate in place |

Implementation cost is large but bounded — Phase A alone is one
session, Phase B is two, full set is roughly 3 months of focused work.
None of it is research-risky: the architecture (define-then-run +
graph-to-graph autograd + live `egui`) is already in place.

## Planned for v1.0 — Production ready

- **Model zoo.** ResNet, MobileNet, GPT-2 small, ViT — loadable with
  one function call.
- **HuggingFace weight loader.** SafeTensors → RustyASG, plus
  PyTorch-checkpoint converter.
- **Dataset loaders.** MNIST, CIFAR-10/100, ImageNet, common text
  datasets with tokenisers.
- **ONNX export.** `asg_to_onnx()` round-trip.
- **Multi-GPU / distributed.** Data-parallel then model-parallel.
- **WebAssembly target.** Browser-resident training and inference.
- **Profiling and debugging tooling.** Tensor inspection, memory
  profiler, operation timing, path highlighting in the visualiser.

---

*Last updated: April 2026 (v0.4.1).*
