# RustyASG Roadmap

This document describes the development plan for the RustyASG project and current implementation progress.

## Current Status

RustyASG is in active development. Most basic components of a production-ready deep learning framework have been implemented.

### Implemented Components

#### Framework Core
- [x] ASG (Abstract Semantic Graph) architecture
- [x] Define-then-run execution model
- [x] Tensor API
- [x] CPU Backend (ndarray)
- [x] GPU Backend (wgpu/WebGPU)
- [x] Shape Inference and static analysis
- [x] Interactive graph visualization (egui)

#### Automatic Differentiation
- [x] Graph-to-graph autograd
- [x] Support for basic operations (Add, Sub, Mul, Div, MatMul)
- [x] ReLU and its gradient
- [x] Softmax and its gradient
- [x] Sigmoid, Tanh, Exp, Log, Neg and their gradients
- [x] LeakyReLU, ELU, GELU, SiLU, Softplus, Abs, Clamp gradients
- [x] Power and its gradient
- [x] MaxPool2d and its gradient (via MaxUnpool2d)
- [x] AvgPool2d and its gradient (via AvgUnpool2d)
- [x] Embedding and its gradient (EmbeddingGrad scatter-add)
- [x] LayerNorm autograd (full implementation with LayerNormBackward, LayerNormGradGamma, LayerNormGradBeta)
- [x] Conv2d autograd (backward pass via Conv2dBackwardInput and Conv2dBackwardWeight)
- [x] Transpose and reshape

#### Neural Network Layers (`nn`)
- [x] Linear (fully connected layer)
- [x] LayerNorm (full autograd support)
- [x] BatchNorm
- [x] Dropout and SpatialDropout
- [x] Conv2d (2D convolution with padding, stride, dilation, groups)
- [x] ConvTranspose2d (transposed convolution)
- [x] MaxPool2d
- [x] AvgPool2d
- [x] AdaptiveAvgPool2d (Global Average Pooling)
- [x] Embedding (lookup layer for NLP)

#### Positional Encoding
- [x] Sinusoidal Positional Encoding
- [x] Learned Positional Embedding
- [x] Rotary Position Embeddings (RoPE)
- [x] ALiBi

#### Attention
- [x] Multi-Head Attention with masks (causal, padding)
- [x] Scaled Dot-Product Attention
- [x] Cross-Attention support

#### Activation Functions
- [x] ReLU
- [x] LeakyReLU
- [x] GELU
- [x] SiLU/Swish
- [x] Tanh
- [x] Sigmoid
- [x] ELU
- [x] Softplus
- [x] Softmax

#### Optimizers
- [x] SGD (with momentum, weight decay, Nesterov)
- [x] Adam
- [x] AdamW
- [x] RMSprop

#### Learning Rate Schedulers
- [x] StepLR
- [x] ExponentialLR
- [x] CosineAnnealingLR
- [x] LinearWarmupLR
- [x] WarmupCosineAnnealingLR

#### Gradient Clipping
- [x] clip_grad_norm
- [x] clip_grad_value

#### Loss Functions
- [x] MSE Loss
- [x] L1 Loss
- [x] Smooth L1 / Huber Loss
- [x] Cross Entropy Loss (with label smoothing)
- [x] Binary Cross Entropy
- [x] BCE with Logits
- [x] KL Divergence
- [x] NLL Loss
- [x] Hinge Loss
- [x] Focal Loss
- [x] Cosine Embedding Loss
- [x] Triplet Margin Loss
- [x] Margin Ranking Loss

#### Serialization
- [x] SafeTensors format (save/load)
- [x] Checkpoint system (weights + optimizer state + metadata)
- [x] CheckpointManager (automatic rotation)

#### Data Pipeline
- [x] Dataset trait and InMemoryDataset
- [x] MapDataset (lazy transforms)
- [x] ConcatDataset
- [x] SubsetDataset
- [x] train_test_split
- [x] DataLoader with batching
- [x] Samplers (Sequential, Random, Weighted, Batch)
- [x] Transforms (Normalize, MinMaxScale, OneHot, Clip, Log, Flatten, RandomNoise)

#### Metrics
- [x] Classification: Accuracy, Precision, Recall, F1Score
- [x] Confusion Matrix (Binary and MultiClass)
- [x] TopKAccuracy
- [x] Regression: MSE, RMSE, MAE, R², MAPE, ExplainedVariance, MaxError
- [x] Running statistics (RunningMean, RunningStd, EMA)
- [x] MetricLogger
- [x] EarlyStopping

---

## Development Plan

### Phase 1: Critical Fixes (High Priority)

#### 1.1 ~~Fix LayerNorm autograd~~ ✅ DONE
**Status:** Completed
**Description:**
Implemented specialized NodeType operations for correct autograd:
- `LayerNorm` - forward pass
- `LayerNormBackward` - gradient w.r.t. input x considering dependencies through mean and variance
- `LayerNormGradGamma` - gradient w.r.t. gamma parameter
- `LayerNormGradBeta` - gradient w.r.t. beta parameter

All gradient check tests pass.

#### 1.2 ~~Conv2d autograd~~ ✅ DONE
**Status:** Completed
**Description:**
Implemented backward operations for Conv2d:
- `Conv2dBackwardInput` - gradient w.r.t. input (transposed convolution)
- `Conv2dBackwardWeight` - gradient w.r.t. weights

All gradient check tests pass (5 tests: basic, with_padding, with_stride, multi_channel, input).

---

### Phase 2: Convolution Operations (Medium Priority)

#### 2.1 Conv2d ✅ DONE
**Status:** Completed
**Description:**
2D convolution fully implemented with autograd support.

**Implemented:**
- [x] NodeType::Conv2d with parameters (kernel_size, stride, padding, dilation, groups)
- [x] CPU implementation (im2col + matmul)
- [x] Autograd for Conv2d (Conv2dBackwardInput, Conv2dBackwardWeight)
- [ ] GPU implementation (WGSL shader)
- [ ] Depthwise and Grouped convolutions

#### 2.2 Pooling Operations
**Status:** Pending
**Complexity:** Medium

**Tasks:**
- [ ] MaxPool2d (with index return for backward)
- [ ] AvgPool2d
- [ ] AdaptiveAvgPool2d
- [ ] AdaptiveMaxPool2d
- [ ] GlobalAveragePooling

#### 2.3 Transposed Convolution
**Status:** Pending
**Complexity:** Medium

**Tasks:**
- [ ] ConvTranspose2d for decoders and generative models

---

### Phase 3: Transformer Support Extension

#### 3.1 Multi-Head Attention
**Status:** Pending
**Complexity:** Medium
**Description:**
Full MHA implementation with masks.

**Tasks:**
- [ ] Scaled Dot-Product Attention
- [ ] Multi-Head Attention with projections
- [ ] Attention masks (causal, padding)
- [ ] Flash Attention optimization (GPU)

#### 3.2 Positional Encoding
**Status:** In Progress

**Tasks:**
- [x] Sinusoidal Positional Encoding
- [x] Learned Positional Embeddings
- [ ] Rotary Position Embeddings (RoPE)
- [ ] ALiBi

#### 3.3 Embedding Layers
**Status:** Done ✅

**Tasks:**
- [x] nn::Embedding
- [ ] nn::EmbeddingBag

---

### Phase 4: Performance Optimizations

#### 4.1 Kernel Fusion
**Status:** Pending
**Complexity:** High
**Description:**
Combining sequential operations into a single WGSL shader.

**Tasks:**
- [ ] Pattern detection for fusable operations
- [ ] Code generation for fused kernels
- [ ] Bias + Activation fusion
- [ ] LayerNorm fusion

#### 4.2 Memory Management
**Status:** Pending
**Complexity:** Medium

**Tasks:**
- [ ] GPU Buffer pooling
- [ ] Memory reuse analysis
- [ ] Gradient checkpointing
- [ ] Mixed precision support (f16)

#### 4.3 Parallelization
**Status:** Pending

**Tasks:**
- [ ] Data parallelism (multi-GPU)
- [ ] Async data loading
- [ ] Pipeline parallelism

---

### Phase 5: Ecosystem Extension

#### 5.1 Model Zoo
**Status:** Pending

**Tasks:**
- [ ] MLP
- [ ] CNN (LeNet, ResNet blocks)
- [ ] Transformer Encoder/Decoder
- [ ] GPT-style model
- [ ] Vision Transformer (ViT)

#### 5.2 Pre-trained Models
**Status:** Pending

**Tasks:**
- [ ] Load weights from HuggingFace (SafeTensors)
- [ ] Converter from PyTorch checkpoint
- [ ] Model Hub integration

#### 5.3 Datasets
**Status:** Pending

**Tasks:**
- [ ] MNIST loader
- [ ] CIFAR-10/100 loader
- [ ] ImageNet loader
- [ ] Text datasets (tokenization)

---

### Phase 6: Developer Experience

#### 6.1 Visualizer Improvements
**Status:** Pending

**Tasks:**
- [ ] Toggle between forward/gradient graphs
- [ ] Detailed node information (hover)
- [ ] Path highlighting
- [ ] Export graph to PNG/SVG
- [ ] Node search

#### 6.2 Debugging and Profiling
**Status:** Pending

**Tasks:**
- [ ] Tensor value inspection
- [ ] Gradient checking utilities
- [ ] Memory profiler
- [ ] Performance profiler
- [ ] Operation timing

#### 6.3 Documentation
**Status:** Pending

**Tasks:**
- [ ] API documentation (rustdoc)
- [ ] Tutorials
- [ ] Examples (MNIST, Text classification)
- [ ] Architecture guide

---

### Phase 7: Production Features

#### 7.1 Model Export
**Status:** Pending

**Tasks:**
- [ ] ONNX export
- [ ] TorchScript-like serialization
- [ ] Inference-only mode (without autograd overhead)

#### 7.2 Deployment
**Status:** Pending

**Tasks:**
- [ ] WebAssembly target
- [ ] Mobile support (iOS/Android via wgpu)
- [ ] Server deployment utilities

#### 7.3 Distributed Training
**Status:** Future

**Tasks:**
- [ ] Distributed data parallel
- [ ] Gradient synchronization
- [ ] Model sharding

---

## Priorities

### Critical (blocking production use)
1. ~~LayerNorm autograd fix~~ ✅ DONE
2. ~~Conv2d operation~~ ✅ DONE
3. Improved error messages (in progress)

### High (significantly improve usability)
1. ~~Multi-Head Attention~~ ✅ DONE
2. Model checkpointing improvements
3. Better documentation

### Medium (nice to have)
1. Kernel fusion
2. Model zoo
3. Visualizer improvements

### Low (future)
1. Distributed training
2. WebAssembly deployment
3. Mobile support

---

## Comparison with Competitors

| Feature | RustyASG | Burn | Candle | PyTorch |
|---------|----------|------|--------|---------|
| Language | Rust | Rust | Rust | Python/C++ |
| Graph Model | Define-then-run | Eager | Eager | Both |
| GPU Backend | wgpu | wgpu/CUDA | CUDA | CUDA |
| Visualization | Built-in | No | No | TensorBoard |
| Autograd | Graph-to-graph | Tape | Tape | Tape |
| SafeTensors | Yes | Yes | Yes | Yes |
| Transformers | Partial | Yes | Yes | Yes |
| Conv2d | Planned | Yes | Yes | Yes |
| Distributed | No | Yes | No | Yes |
| WebAssembly | Planned | Yes | No | No |

### Unique Advantages of RustyASG
1. **Built-in graph visualization** - unique debugging capability
2. **Define-then-run** - enables global optimizations
3. **Pure Rust architecture** - no Python/C++ dependencies
4. **Educational value** - well-structured code

---

## Contributing

We welcome contributions to the project! Especially valuable are:

1. **Bug reports** - especially related to autograd
2. **Performance improvements** - GPU optimizations
3. **New operations** - Conv2d, Attention, etc.
4. **Documentation** - examples, tutorials
5. **Testing** - gradient checking, edge cases

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## Timeline

| Milestone | Target | Status |
|-----------|--------|--------|
| v0.1 - Core | Done | ✅ |
| v0.2 - Training Utils | Done | ✅ |
| v0.3 - Data Pipeline | Done | ✅ |
| v0.4 - Metrics | Done | ✅ |
| v0.5 - Conv2d | Q1 2026 | Planned |
| v0.6 - Transformers | Q2 2026 | Planned |
| v1.0 - Production Ready | Q4 2026 | Planned |

---

*Last updated: January 2026*
