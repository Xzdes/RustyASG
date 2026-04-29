# Publishing RustyASG v0.4.1

This document is the release runbook for **rustyasg v0.4.1** —
**Phase A of the Interactive Model Lab (read-only inspection)**.

The library public API is unchanged compared to v0.4.0; this release is
purely additive (all changes ship in the binary and the `gui_viewer`
module). Anyone who depends on the library API can upgrade `0.4.0 →
0.4.1` without code changes.

---

## What was done in this release

### Phase A — Interactive Model Lab (read-only)

The live `egui` graph viewer is no longer just a structure renderer. It's
now a real diagnostic and teaching tool:

- **Educational Node Inspector.** Click any node → side panel explains, in
  plain English or Russian:
  - **What this node does** — one-paragraph plain-language description.
  - **Formula** — the math written in monospace
    (e.g. `softmax(xᵢ) = eˣⁱ / Σⱼ eˣʲ`,
    `y = γ · (x − μ)/√(σ² + ε) + β`).
  - **Why it's used** — role in real architectures (GELU → "the default
    FFN activation in modern transformers (BERT, GPT-2, ViT)";
    MatMul → "the single most expensive op in deep nets…").
  - **Role in this model** — for `Parameter` nodes, the inspector parses
    the parameter name (e.g. `transformer.norm1.gamma` → LayerNorm
    scale γ; `mha.w_q` → Multi-Head Attention query projection;
    `fc1.weights` → linear weight matrix) and explains the role plus the
    initialisation strategy.
  - **Graph-output marker** when the node is a graph output.
  - Technical fields (`id`, `name`, `type`, `shape`, `dtype`, `inputs`)
    moved to a collapsible "Technical details" section at the bottom.
- **Live loss chart** in a docked bottom panel, auto-rescales on
  min/max as the compute thread emits `EpochDone`.
- **Edge highlighting** — the selected node's incident edges are drawn
  in amber so dataflow is obvious.
- **Per-category color coding** — input / parameter / literal /
  arithmetic / activation / reduction / normalisation / convolution /
  pooling / shape op / gradient op each get a distinct fill colour;
  graph outputs get a brighter peach.
- **Two-language UI.** New `--lang en|ru` CLI flag selects English
  (default) or Russian for every label and description.
- **`ComputeUpdate` channel protocol.** The compute thread now sends a
  typed enum (`GraphReady` / `EpochDone`) over `mpsc` instead of a raw
  `Asg`. This is the foundation for future Phase B mutations.

### Coverage of the educational descriptions

Every `NodeType` produced by current layers has a dedicated description:

- **Data:** `Input`, `Parameter`, `Literal`, `External`.
- **Arithmetic:** `Add`, `Subtract`, `Multiply`, `Divide`,
  `MatrixMultiply`, `Power`, `GreaterThan`.
- **Activations:** `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `SiLU`,
  `LeakyReLU`, `ELU`, `Softplus`, `Softmax`, `Clamp`, `Abs`, `Neg`,
  `Exp`, `Log`, `Sqrt`.
- **Reductions:** `Sum`, `Mean`, `Variance`, `MeanAxis`,
  `VarianceAxis`.
- **Normalisation:** `LayerNorm`, `BatchNorm` and all four backward
  variants for each (`*Backward`, `*GradGamma`, `*GradBeta`).
- **Convolutions:** `Conv2d`, `ConvTranspose2d`, `Conv2dBackwardInput`,
  `Conv2dBackwardWeight`.
- **Pooling:** `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`,
  `MaxUnpool2d`, `AvgUnpool2d`.
- **Embedding:** `Embedding`, `EmbeddingGrad`.
- **Shape ops:** `Reshape`, `Transpose`, `Slice`, `Concat`, `Broadcast`,
  `ReduceSumTo`, `SliceBackward`, `DropoutMask`.
- **Control flow:** `If`, `ForLoop`, `FunctionDefinition`,
  `FunctionCall`, `Print`.

### Bumped

- `0.4.0` → `0.4.1`. Purely additive release.

### Bonus housekeeping

- Updated `Cargo.lock` to bump `immutable-chunkmap v2.1.0 → v2.1.2`
  (transitive dep through `eframe` → `accesskit`); the previous
  pinned version was yanked from crates.io.

---

## Pre-flight checks (all green)

Run these in order. Each one already passed locally on `main`; they're
listed here so the steps are reproducible.

```bash
# 1. Format clean.
cargo fmt --all --check                       # exit 0

# 2. Strict clippy across lib + bin + tests + examples.
cargo clippy --release --all-targets -- -D warnings

# 3. Library unit tests.
cargo test --release --lib                    # 93 passed

# 4. Numerical gradient checks.
cargo test --release --test grad_check        # 9 passed (2 documented ignores)

# 5. Optional: GPU↔CPU parity tests (requires a GPU adapter).
cargo test --release --test gpu_backend -- --test-threads=1   # 48 passed

# 6. Strict docs build.
RUSTDOCFLAGS="-D rustdoc::broken_intra_doc_links" cargo doc --release --no-deps

# 7. Package preview (compiles into target/package/rustyasg-0.4.1/).
cargo package --allow-dirty --no-verify       # 61 files, 217.6 KiB compressed
```

**Total local test count: 150 passing + 2 documented `#[ignore]`.**

---

## Publishing to crates.io

> Make sure you've already run `cargo login` once with a valid crates.io
> API token. If not: https://crates.io/me/ → API Tokens → New Token →
> `cargo login <token>`.

### 1. Final commit on `main`

```bash
git status                # only Cargo.toml, Cargo.lock, CHANGELOG.md,
                          # ROADMAP.md, README.md, README.ru.md,
                          # src/main.rs, src/gui_viewer.rs, plus this file

git add -A
git commit -m "Release v0.4.1 — Phase A: Interactive Model Lab"
git push origin main
```

### 2. Tag the release

```bash
git tag -a v0.4.1 -m "v0.4.1 — Phase A: Interactive Model Lab (read-only)"
git push origin v0.4.1
```

### 3. Publish

Dry run first (catches metadata problems before the real upload):

```bash
cargo publish --dry-run
```

If clean, publish for real:

```bash
cargo publish
```

This uploads `target/package/rustyasg-0.4.1.crate` (217.6 KiB
compressed) to crates.io.

### 4. Verify

- **crates.io:** https://crates.io/crates/rustyasg — should show
  `0.4.1` within a minute.
- **docs.rs:** https://docs.rs/rustyasg/0.4.1 — build kicks off
  automatically; usually finishes in 5–10 min.
- **cargo install:** `cargo install rustyasg --version 0.4.1` should
  work from a clean cache after the index updates.
- **End user smoke test:**
  ```bash
  cargo new testapp && cd testapp
  echo 'rustyasg = "0.4.1"' >> Cargo.toml
  # Add a couple of lines that use rustyasg::nn::Linear, rustyasg::tensor.
  cargo build
  ```

### 5. GitHub release

After the tag is pushed, GitHub will show the tag at
https://github.com/Xzdes/RustyAsg/releases. Click "Draft a new release"
on `v0.4.1` and paste the release notes:

```markdown
# v0.4.1 — Phase A: Interactive Model Lab (read-only)

This release lands **Phase A** of the long-planned **Interactive Model
Lab**: the live `egui` graph viewer is no longer just a structure
renderer — it's a real diagnostic and teaching tool. Library API is
unchanged (purely additive); only the binary and the visualiser gained
features.

## Highlights

- **Educational Node Inspector.** Click any node → side panel explains
  *what* the operation does, the *formula*, *why* it's used, and (for
  parameters) its *role* in this specific model. Plain English or
  Russian, selected at startup with `--lang en|ru`.
- **Live loss chart.** Bottom panel auto-renders an XY plot of training
  loss vs. epoch.
- **Edge highlighting + per-category color coding.** Selected node's
  incident edges get amber highlight; nodes are filled by category.
- **`ComputeUpdate` channel protocol.** Typed `GraphReady` /
  `EpochDone` enum replaces the raw `Asg` send; foundation for Phase B
  mutations.

## Coverage

Every `NodeType` produced by current layers has a dedicated educational
description: data nodes, arithmetic, activations (`ReLU`, `GELU`,
`Softmax`, …), reductions, normalisation (`LayerNorm`, `BatchNorm` +
all backward variants), convolutions (`Conv2d` / `ConvTranspose2d` +
backwards), pooling, embedding, shape ops, control flow.

## Compatibility

- Library public API: **unchanged** from v0.4.0.
- Binary: new `--lang en|ru` flag.
- MSRV: 1.75 (unchanged).

See [CHANGELOG.md](CHANGELOG.md) for details.
```

### 6. Announcement (optional)

- **r/rust** weekly thread on Sunday — short post linking to crates.io,
  GitHub repo, and the docs.
- **This Week in Rust** — submit a PR to the `submissions/` directory.
- **Rust Discord** — `#announcements` channel.

---

## Rollback

If a problem surfaces post-publish:

1. **Yank** (does not delete, but blocks new dependents from picking
   the version):
   ```bash
   cargo yank --version 0.4.1 rustyasg
   ```
2. **Un-yank** if the issue is resolved by a follow-up:
   ```bash
   cargo yank --version 0.4.1 rustyasg --undo
   ```
3. **Hotfix:** bump to `0.4.2`, fix the issue, repeat the publish flow.

> Never publish the same version twice; once a `.crate` is up, the
> tarball is immutable. To replace, ship a new version.

---

## Post-release housekeeping

- Update `[Unreleased]` in `CHANGELOG.md` with anything that landed
  after the cut (currently empty — only deferred / planned items).
- Push README badges if anything changed (the `Crates.io` badge auto-
  updates from crates.io).
- Update the issue tracker: close any issues marked
  `target: v0.4.1`; move open `target: v0.4.1` items to `v0.5`.
- File the next milestone (v0.5 — Performance & production) on GitHub
  if it doesn't exist yet.

---

*Last updated: April 2026, for v0.4.1.*
