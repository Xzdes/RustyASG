# RustyASG Roadmap

Этот документ описывает план развития проекта RustyASG и текущий прогресс реализации.

## Текущий статус

RustyASG находится в активной разработке. Большинство базовых компонентов production-ready deep learning фреймворка уже реализованы.

### Реализованные компоненты

#### Ядро фреймворка
- [x] ASG (Abstract Semantic Graph) архитектура
- [x] Define-then-run execution model
- [x] Tensor API
- [x] CPU Backend (ndarray)
- [x] GPU Backend (wgpu/WebGPU)
- [x] Shape Inference и статический анализ
- [x] Интерактивная визуализация графа (egui)

#### Автоматическое дифференцирование
- [x] Граф-в-граф autograd
- [x] Поддержка базовых операций (Add, Sub, Mul, Div, MatMul)
- [x] ReLU и его градиент
- [x] Softmax и его градиент
- [x] Sigmoid, Tanh, Exp, Log, Neg и их градиенты
- [x] LeakyReLU, ELU, GELU, SiLU, Softplus, Abs, Clamp градиенты
- [x] Power и его градиент
- [x] MaxPool2d и его градиент (через MaxUnpool2d)
- [x] AvgPool2d и его градиент (через AvgUnpool2d)
- [x] Embedding и его градиент (EmbeddingGrad scatter-add)
- [x] LayerNorm autograd (полная реализация с LayerNormBackward, LayerNormGradGamma, LayerNormGradBeta)
- [x] Conv2d autograd (backward pass через Conv2dBackwardInput и Conv2dBackwardWeight)
- [x] Транспонирование и reshape

#### Нейросетевые слои (`nn`)
- [x] Linear (полносвязный слой)
- [x] LayerNorm (полная поддержка autograd)
- [x] BatchNorm
- [x] Dropout и SpatialDropout
- [x] Conv2d (2D свертка с padding, stride, dilation, groups)
- [x] ConvTranspose2d (транспонированная свертка)
- [x] MaxPool2d
- [x] AvgPool2d
- [x] AdaptiveAvgPool2d (Global Average Pooling)
- [x] Embedding (lookup слой для NLP)

#### Positional Encoding
- [x] Sinusoidal Positional Encoding
- [x] Learned Positional Embedding
- [x] Rotary Position Embeddings (RoPE)
- [x] ALiBi

#### Attention
- [x] Multi-Head Attention с масками (causal, padding)
- [x] Scaled Dot-Product Attention
- [x] Cross-Attention support

#### Функции активации
- [x] ReLU
- [x] LeakyReLU
- [x] GELU
- [x] SiLU/Swish
- [x] Tanh
- [x] Sigmoid
- [x] ELU
- [x] Softplus
- [x] Softmax

#### Оптимизаторы
- [x] SGD (с momentum, weight decay, Nesterov)
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

#### Loss функции
- [x] MSE Loss
- [x] L1 Loss
- [x] Smooth L1 / Huber Loss
- [x] Cross Entropy Loss (с label smoothing)
- [x] Binary Cross Entropy
- [x] BCE with Logits
- [x] KL Divergence
- [x] NLL Loss
- [x] Hinge Loss
- [x] Focal Loss
- [x] Cosine Embedding Loss
- [x] Triplet Margin Loss
- [x] Margin Ranking Loss

#### Сериализация
- [x] SafeTensors формат (сохранение/загрузка)
- [x] Checkpoint система (веса + optimizer state + metadata)
- [x] CheckpointManager (автоматическая ротация)

#### Data Pipeline
- [x] Dataset trait и InMemoryDataset
- [x] MapDataset (lazy transforms)
- [x] ConcatDataset
- [x] SubsetDataset
- [x] train_test_split
- [x] DataLoader с batching
- [x] Samplers (Sequential, Random, Weighted, Batch)
- [x] Transforms (Normalize, MinMaxScale, OneHot, Clip, Log, Flatten, RandomNoise)

#### Metrics
- [x] Classification: Accuracy, Precision, Recall, F1Score
- [x] Confusion Matrix (Binary и MultiClass)
- [x] TopKAccuracy
- [x] Regression: MSE, RMSE, MAE, R², MAPE, ExplainedVariance, MaxError
- [x] Running statistics (RunningMean, RunningStd, EMA)
- [x] MetricLogger
- [x] EarlyStopping

---

## План развития

### Фаза 1: Критические исправления (High Priority)

#### 1.1 ~~Исправить LayerNorm autograd~~ ✅ DONE
**Статус:** Завершено
**Описание:**
Реализованы специализированные NodeType операции для корректного autograd:
- `LayerNorm` - forward pass
- `LayerNormBackward` - градиент по входу x с учётом зависимостей через mean и variance
- `LayerNormGradGamma` - градиент по параметру gamma
- `LayerNormGradBeta` - градиент по параметру beta

Все gradient check тесты проходят.

#### 1.2 ~~Conv2d autograd~~ ✅ DONE
**Статус:** Завершено
**Описание:**
Реализованы backward операции для Conv2d:
- `Conv2dBackwardInput` - градиент по входу (транспонированная свертка)
- `Conv2dBackwardWeight` - градиент по весам

Все gradient check тесты проходят (5 тестов: basic, with_padding, with_stride, multi_channel, input).

---

### Фаза 2: Сверточные операции (Medium Priority)

#### 2.1 Conv2d ✅ DONE
**Статус:** Завершено
**Описание:**
2D свертка полностью реализована с поддержкой autograd.

**Реализовано:**
- [x] NodeType::Conv2d с параметрами (kernel_size, stride, padding, dilation, groups)
- [x] CPU реализация (im2col + matmul)
- [x] Autograd для Conv2d (Conv2dBackwardInput, Conv2dBackwardWeight)
- [ ] GPU реализация (WGSL шейдер)
- [ ] Depthwise и Grouped convolutions

#### 2.2 Pooling операции
**Статус:** Pending
**Сложность:** Средняя

**Задачи:**
- [ ] MaxPool2d (с возвратом индексов для backward)
- [ ] AvgPool2d
- [ ] AdaptiveAvgPool2d
- [ ] AdaptiveMaxPool2d
- [ ] GlobalAveragePooling

#### 2.3 Transposed Convolution
**Статус:** Pending
**Сложность:** Средняя

**Задачи:**
- [ ] ConvTranspose2d для декодеров и генеративных моделей

---

### Фаза 3: Расширение Transformer поддержки

#### 3.1 Multi-Head Attention
**Статус:** Pending
**Сложность:** Средняя
**Описание:**
Полноценная реализация MHA с масками.

**Задачи:**
- [ ] Scaled Dot-Product Attention
- [ ] Multi-Head Attention с проекциями
- [ ] Attention masks (causal, padding)
- [ ] Flash Attention оптимизация (GPU)

#### 3.2 Positional Encoding
**Статус:** In Progress

**Задачи:**
- [x] Sinusoidal Positional Encoding
- [x] Learned Positional Embeddings
- [ ] Rotary Position Embeddings (RoPE)
- [ ] ALiBi

#### 3.3 Embedding слои
**Статус:** Done ✅

**Задачи:**
- [x] nn::Embedding
- [ ] nn::EmbeddingBag

---

### Фаза 4: Оптимизации производительности

#### 4.1 Kernel Fusion
**Статус:** Pending
**Сложность:** Высокая
**Описание:**
Объединение последовательных операций в один WGSL шейдер.

**Задачи:**
- [ ] Pattern detection для fusable операций
- [ ] Code generation для fused kernels
- [ ] Bias + Activation fusion
- [ ] LayerNorm fusion

#### 4.2 Memory Management
**Статус:** Pending
**Сложность:** Средняя

**Задачи:**
- [ ] GPU Buffer pooling
- [ ] Memory reuse analysis
- [ ] Gradient checkpointing
- [ ] Mixed precision support (f16)

#### 4.3 Parallelization
**Статус:** Pending

**Задачи:**
- [ ] Data parallelism (multi-GPU)
- [ ] Async data loading
- [ ] Pipeline parallelism

---

### Фаза 5: Расширение экосистемы

#### 5.1 Model Zoo
**Статус:** Pending

**Задачи:**
- [ ] MLP
- [ ] CNN (LeNet, ResNet blocks)
- [ ] Transformer Encoder/Decoder
- [ ] GPT-style model
- [ ] Vision Transformer (ViT)

#### 5.2 Pre-trained Models
**Статус:** Pending

**Задачи:**
- [ ] Загрузка весов из HuggingFace (SafeTensors)
- [ ] Конвертер из PyTorch checkpoint
- [ ] Model Hub интеграция

#### 5.3 Datasets
**Статус:** Pending

**Задачи:**
- [ ] MNIST loader
- [ ] CIFAR-10/100 loader
- [ ] ImageNet loader
- [ ] Text datasets (tokenization)

---

### Фаза 6: Developer Experience

#### 6.1 Улучшения визуализатора
**Статус:** Pending

**Задачи:**
- [ ] Переключение между forward/gradient графами
- [ ] Подробная информация об узлах (hover)
- [ ] Подсветка путей
- [ ] Экспорт графа в PNG/SVG
- [ ] Поиск узлов

#### 6.2 Debugging и Profiling
**Статус:** Pending

**Задачи:**
- [ ] Tensor value inspection
- [ ] Gradient checking utilities
- [ ] Memory profiler
- [ ] Performance profiler
- [ ] Operation timing

#### 6.3 Документация
**Статус:** Pending

**Задачи:**
- [ ] API documentation (rustdoc)
- [ ] Tutorials
- [ ] Examples (MNIST, Text classification)
- [ ] Architecture guide

---

### Фаза 7: Production Features

#### 7.1 Model Export
**Статус:** Pending

**Задачи:**
- [ ] ONNX export
- [ ] TorchScript-like serialization
- [ ] Inference-only mode (без autograd overhead)

#### 7.2 Deployment
**Статус:** Pending

**Задачи:**
- [ ] WebAssembly target
- [ ] Mobile support (iOS/Android через wgpu)
- [ ] Server deployment utilities

#### 7.3 Distributed Training
**Статус:** Future

**Задачи:**
- [ ] Distributed data parallel
- [ ] Gradient synchronization
- [ ] Model sharding

---

## Приоритеты

### Критические (блокируют production use)
1. ~~LayerNorm autograd fix~~ ✅ DONE
2. ~~Conv2d операция~~ ✅ DONE
3. Improved error messages (in progress)

### Высокие (значительно улучшат usability)
1. ~~Multi-Head Attention~~ ✅ DONE
2. Model checkpointing improvements
3. Better documentation

### Средние (nice to have)
1. Kernel fusion
2. Model zoo
3. Visualizer improvements

### Низкие (future)
1. Distributed training
2. WebAssembly deployment
3. Mobile support

---

## Сравнение с конкурентами

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

### Уникальные преимущества RustyASG
1. **Встроенная визуализация графа** - уникальная возможность для отладки
2. **Define-then-run** - позволяет глобальные оптимизации
3. **Чистая Rust архитектура** - без Python/C++ зависимостей
4. **Образовательная ценность** - хорошо структурированный код

---

## Contributing

Мы приветствуем вклад в развитие проекта! Особенно ценны:

1. **Bug reports** - особенно связанные с autograd
2. **Performance improvements** - GPU оптимизации
3. **New operations** - Conv2d, Attention, etc.
4. **Documentation** - примеры, туториалы
5. **Testing** - gradient checking, edge cases

См. [CONTRIBUTING.md](CONTRIBUTING.md) для деталей.

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

*Последнее обновление: Январь 2026*
