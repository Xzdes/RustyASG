# Changelog

All notable changes to RustyASG are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Deferred to v0.5 (performance-focused)
- [ ] GPU buffer pool — reuse allocations between training steps.
- [ ] Kernel fusion: `MatMul + Bias + Activation` in a single WGSL kernel.
- [ ] Mixed precision (f16) on GPU.
- [ ] Inference-only mode API (no autograd overhead).
- [ ] Criterion benchmarks against CPU/GPU.
- [ ] Replace remaining `unwrap()`s in critical paths with typed `RustyAsgError`
      (~125 call sites — most are inside already-`Result`-returning functions
      and guarded by validated invariants).
- [ ] Tiny GPT block (needs causal masking + multi-batch inputs).
- [ ] Vision Transformer (ViT) starter.

## [0.3.1] - 2026-04-24 (pre-release polish)

### Added
- **CNN classifier example** (`examples/cnn_classifier.rs`): end-to-end demo of
  Conv2d + AvgPool + AdaptiveAvgPool + Linear + Adam, reaching 100% accuracy
  on a 3-class synthetic dataset. First example to exercise the full CNN
  autograd stack (`Conv2dBackward*`, `AvgUnpool2d`).
- **`docs.rs` integration**: `[package.metadata.docs.rs]` section in
  `Cargo.toml` with `--cfg docsrs`.
- **Release profile tuning**: `opt-level = 3`, `lto = "thin"`,
  `codegen-units = 1`, `strip = "debuginfo"`.
- **Full CI pipeline**: dedicated jobs for `cargo fmt --check`,
  `cargo clippy --all-targets -- -D warnings`, strict `cargo doc`, and
  cross-platform `cargo test` matrix (Linux/Windows/macOS).

### Changed
- **`cargo fmt` applied across the entire tree** (28 files reformatted).
- **`cargo clippy -- -D warnings` clean for all targets**: library, tests,
  binary, and examples. Library suppresses only three design-motivated lints
  at crate level (`too_many_arguments`, `type_complexity`,
  `should_implement_trait`); examples disable `needless_range_loop` and
  `if_same_then_else` as demo-style allowances.
- **Rustdoc**: every shape notation like `[N, C_in, H, W]` escaped in code
  blocks so `RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links"` is clean.
  Fixes 10 previously-broken intra-doc links.
- **`train_test_split`**: the `D: Dataset + Clone` bound is consolidated into
  one `where` clause to avoid `clippy::multiple_bound_locations`.
- **Cargo.toml**: expanded crate metadata (rust-version, homepage,
  documentation URL, `exclude` list to keep published crate small, explicit
  `[lib]`/`[bin]` targets).

### Removed
- `logo.png` (1.2 MB) from the published crate via `exclude`. Still present
  in the repo for README rendering on GitHub.

## [0.3.0] - 2026-04-24

### Added
- **Declarative layer API (`ParameterRegistry`).** Every `nn::*` layer now
  registers its parameter shapes and initializers with `GraphContext`:
  ```rust
  let fc = Linear::new(&ctx, "fc1", 784, 128);  // shapes auto-registered
  ctx.borrow().init_parameters(&mut runtime_data);  // Xavier / Zeros sampled
  ShapeInference::run_with_context(&mut graph, &ctx.borrow(), &input_shapes)?;
  ```
- New `nn::init` module with 9 standard initializers: `Zeros`, `Ones`,
  `Constant`, `Uniform`, `Normal`, `XavierUniform`, `XavierNormal`,
  `KaimingUniform`, `KaimingNormal`.
- `Tensor::new_parameter_with_shape(ctx, name, shape, init)` — preferred
  constructor for trainable weights.
- `GraphContext::{register_parameter_meta, parameter_meta, parameter_registry,
  build_shape_map, init_parameters}`.
- `ShapeInference::run_with_context()` — pulls parameter shapes from the
  registry, so callers only provide input shapes.
- **GPU backend completeness (Phase 3):**
  - `LayerNorm` forward + 3 backward WGSL shaders (`LayerNormBackward`,
    `LayerNormGradGamma`, `LayerNormGradBeta`). Transformers now train
    end-to-end on GPU.
  - `Conv2dBackwardInput` / `Conv2dBackwardWeight` (groups=1, dilation=(1,1)).
  - `MaxPool2d`, `MaxUnpool2d`, `AvgPool2d`, `AvgUnpool2d`, `AdaptiveAvgPool2d`.
  - `Embedding`, `EmbeddingGrad`.
  - `ConvTranspose2d` forward with bias.
  - New `dispatch_rowwise` helper for shaders that parallelize per-row/column.
- **Slice / Concat / SliceBackward primitives.** New NodeTypes with full
  support across CPU, GPU, shape inference, and autograd. Enables real
  tensor-splitting operations in user code.
- **Tensor API additions:** `Tensor::slice(axis, start, end)` and
  `Tensor::concat(others, axis)`.
- **Full RoPE implementation.** `RotaryPositionEmbedding` now uses the
  standard split-half formulation via Slice/Concat — mathematically correct
  and end-to-end differentiable. Previously a stub that added `cos` as bias.
- **`--gpu` CLI flag** for `cargo run` (default is now CPU, opt-in GPU).

### Changed
- **Breaking (migration required for all v0.2 users):** every `nn::*` layer
  constructor now requires dimension arguments:
  - `Linear::new(ctx, name, in_features, out_features)` (was `new(ctx, name)`)
  - `LayerNorm::new(ctx, name, normalized_shape)` (was `new(ctx, name)`)
  - `BatchNorm::new(ctx, name, num_features)` (was `new(ctx, name)`)
  - `FeedForward::new(ctx, name, embed_dim, hidden_dim)` (was `new(ctx, name)`)
  - `Conv2d::new(ctx, name, in, out, kernel)` — now actually persists config.
  - `ConvTranspose2d::new(ctx, name, in, out, kernel)` — now actually persists config.
- `main.rs` no longer uses string-matching to infer parameter shapes
  (`name.contains("w_q")` → gone). Shapes come from the registry.
- `ShapeInference::run_with_context` is the recommended entry point; the old
  `ShapeInference::run` still exists but requires manually supplying every
  parameter shape.

### Fixed
- RoPE no longer silently produces wrong outputs (was adding `cos` as bias).
- `ConvTranspose2d::new` no longer ignores its `in_channels`/`out_channels`/
  `kernel_size` arguments.

### Removed
- Dead code: unused `minus_one` literal in autograd's `Abs` backward.

## [0.2.0] - 2026-01-20

### Phase 1 (Cleanup and consistency)
- `main.rs` refactored to consume the library via `use rustyasg::*` instead of
  duplicating modules (eliminates ~100 false-positive warnings).
- Replaced deprecated `rand::thread_rng` → `rand::rng` in data/sampler/transforms.
- Added `DataLoaderIterator<'_>` lifetime annotation.
- Synchronized README with roadmap (LayerNorm/Conv2d autograd marked done).

### Earlier (pre-history at v0.2)
- ASG (Abstract Semantic Graph) architecture.
- Graph-to-graph autograd.
- CPU backend (ndarray) with full operation coverage.
- GPU backend (wgpu) with basic ops + Conv2d forward.
- `nn` layers: Linear, Conv2d/Transpose, Pooling, LayerNorm, BatchNorm,
  Dropout, Embedding, MHA, TransformerBlock, FeedForward.
- Positional encodings: Sinusoidal, Learned, RoPE (stub — fixed in 0.3),
  ALiBi.
- 4 optimizers: SGD, Adam, AdamW, RMSprop.
- 5 LR schedulers.
- Gradient clipping.
- 14 loss functions.
- Data pipeline: Dataset, DataLoader, Samplers, Transforms.
- Metrics: classification + regression.
- SafeTensors serialization + checkpointing.
- Interactive graph visualizer (egui).

[Unreleased]: https://github.com/Xzdes/RustyAsg/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Xzdes/RustyAsg/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/Xzdes/RustyAsg/releases/tag/v0.2.0
