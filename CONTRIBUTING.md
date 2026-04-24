# Contributing to RustyASG

Thanks for your interest in RustyASG! This document explains how to build the
project, where tests live, and the conventions we follow so contributions land
smoothly.

## Quick start

```bash
# Clone and build
git clone https://github.com/Xzdes/RustyAsg.git
cd RustyAsg
cargo build --release

# Run everything (library tests + grad checks + GPU sanity tests)
cargo test --release

# Run one of the examples
cargo run --release --example xor

# Train the bundled TransformerBlock demo (CPU default, GPU via --gpu)
cargo run --release
cargo run --release -- --gpu
```

## Project layout

```
src/
├── asg/              # Core ASG data model (NodeType, Asg, etc.)
├── analysis/         # Shape inference and static graph analysis
├── autograd/         # Graph-to-graph reverse-mode autodiff
├── runtime/
│   ├── backend.rs    # Backend trait common to CPU and GPU
│   ├── cpu_backend.rs   # Reference implementation (ndarray)
│   └── wgpu_backend.rs  # GPU implementation (WGSL shaders)
├── nn/               # Layer library (Linear, Conv2d, Attention, ...)
│   └── init.rs       # Weight initializers (Xavier, Kaiming, ...)
├── optimizers/       # SGD, Adam, AdamW, RMSprop + LR schedulers
├── losses.rs         # 14 loss functions
├── data/             # Dataset, DataLoader, samplers, transforms
├── metrics/          # Classification + regression metrics
├── serialization/    # SafeTensors + checkpoint manager
├── tensor.rs         # Symbolic Tensor handle + GraphContext + ParameterRegistry
├── gui_viewer.rs     # Interactive egui graph visualizer (bin only)
└── main.rs           # Demo binary

examples/             # Runnable end-to-end examples (xor, mnist, transformer_classifier, ...)
tests/
├── grad_check.rs     # Numerical gradient verification for autograd
└── gpu_backend.rs    # CPU↔GPU parity tests (the authoritative GPU test suite)
```

## Adding a new operation

Adding a new primitive op touches **four** modules. Skipping any one will
either break compilation or lead to silent runtime errors.

1. **`src/asg/mod.rs`**: add a new `NodeType` variant. Include doc comment
   describing shape invariants.
2. **`src/analysis/shape_inference.rs`**:
   - Add a match arm in `infer_node_shape` that computes the output shape.
   - Add a match arm in `build_sorted_graph` listing the node's data
     dependencies (so topological sort visits them first).
3. **`src/runtime/cpu_backend.rs`**: add the reference implementation in
   `evaluate_node` and a helper `op_xxx` function if it's non-trivial.
4. **`src/runtime/wgpu_backend.rs`**: add a WGSL shader and dispatch logic
   in the big `match` in `run`. Use `dispatch_shader` for one-thread-per-element
   kernels or `dispatch_rowwise` for reduction-style kernels.
5. **`src/autograd/mod.rs`** (only if the op is differentiable):
   - Add a match arm in `backward_node` that emits the gradient sub-graph.
   - Add a match arm in `inputs_of` listing data dependencies (parallel to
     `build_sorted_graph`).
6. **`tests/gpu_backend.rs`**: add a CPU↔GPU parity test using
   `run_graph_and_compare` (or one of the `test_*_op` helpers for pure unary/
   binary ops).

## Adding a new `nn::*` layer

Every layer should register its parameter shapes with the graph context at
construction time. Follow the pattern in [`src/nn/linear.rs`]:

```rust
pub fn new(ctx: &Rc<RefCell<GraphContext>>, name: &str, in_dim: usize, out_dim: usize) -> Self {
    let weights = Tensor::new_parameter_with_shape(
        ctx,
        &format!("{name}.weights"),
        vec![in_dim, out_dim],
        Initializer::XavierUniform,
    );
    // ...
}
```

Initializers live in [`src/nn/init.rs`]. Pick one that matches the layer's
downstream activation (Kaiming for ReLU-like, Xavier for tanh/sigmoid, Normal
for embeddings, etc.).

## Testing conventions

- **Unit tests** live inline in `#[cfg(test)] mod tests` at the bottom of each
  module. They should be fast (< 1 s each) and deterministic.
- **Numerical gradient checks** live in [`tests/grad_check.rs`]. Any new
  backward rule in `autograd` should have one.
- **GPU parity tests** live in [`tests/gpu_backend.rs`]. Compare against
  `CpuBackend` with tolerance `1e-5`. Use `--test-threads=1` when running the
  full GPU suite locally (wgpu does not handle concurrent device creation
  gracefully on all platforms).

Run subsets:

```bash
cargo test --release --lib                    # unit tests only
cargo test --release --test grad_check        # autograd correctness
cargo test --release --test gpu_backend -- --test-threads=1
cargo test --release --test gpu_backend -- --test-threads=1 layer_norm  # filter
```

## Commit & PR style

- One logical change per commit. Descriptive messages — prefer "fix shape
  inference for Slice axis check" over "fix bug".
- Before submitting a PR, run:
  ```bash
  cargo build --release --all-targets
  cargo test --release
  ```
- If your change touches GPU code, run the GPU suite locally (`cargo test
  --release --test gpu_backend -- --test-threads=1`) — CI runners often lack a
  compute-capable adapter and will mark GPU tests as advisory.
- Update [`CHANGELOG.md`] under `[Unreleased]`.
- For behavior-changing PRs, add or adjust a test demonstrating the new
  contract.

## Style

- `cargo fmt` on every PR (advisory in CI today, mandatory at v1.0).
- `cargo clippy --lib` should be warning-free for library code. Tests and
  examples have a grace period through Phase 6.
- Prefer tight, explicit error paths over nested pattern matches.
- Document public items. Leave a one-line comment where a non-obvious
  invariant or workaround lives.

## Reporting issues

If you find a bug, open an issue with:

1. Minimal reproducer (`cargo run --example ...` if possible).
2. Expected vs. actual output.
3. Platform + wgpu backend (Vulkan / DX12 / Metal / OpenGL fallback).
4. Whether CPU and GPU disagree (run the same graph on both if feasible).

Numerical issues are especially valuable when paired with a `grad_check`-style
test case.

## License

By contributing you agree that your contributions will be licensed under the
MIT license of the project.
