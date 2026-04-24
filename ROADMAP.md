# RustyASG Roadmap

This document tracks what is already implemented and where the project is
heading next.

## Current status

RustyASG v0.3.1 is a polished, published-ready crate. The library is
clippy-clean under `-D warnings`, rustdoc builds strictly with
`RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links"`, and the full test
suite (141 tests: 87 lib + 46 GPU + 8 grad check) is green on every
supported platform in CI.

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
- **Tiny GPT block.** End-to-end GPT-style example (needs causal
  masking + multi-batch training).
- **Vision Transformer starter.** Patch embedding + TransformerBlock
  stack.

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

*Last updated: April 2026.*
