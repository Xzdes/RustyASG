// --- Файл: src/losses.rs ---

//! Модуль, содержащий реализации функций потерь в графовой парадигме.
//!
//! Функции потерь здесь - это обычные Rust-функции, которые принимают
//! символьные `Tensor`-дескрипторы и добавляют в граф узлы, необходимые
//! для вычисления значения ошибки.
//!
//! # Доступные функции потерь
//!
//! - **MSE (Mean Squared Error)**: `mse_loss`, `mse_loss_mean`
//! - **L1 (Mean Absolute Error)**: `l1_loss`, `l1_loss_mean`
//! - **Smooth L1 (Huber Loss)**: `smooth_l1_loss`
//! - **Cross-Entropy**: `cross_entropy_loss`, `cross_entropy_with_label_smoothing`
//! - **Binary Cross-Entropy**: `binary_cross_entropy`, `bce_with_logits`
//! - **KL Divergence**: `kl_divergence`
//! - **Negative Log Likelihood**: `nll_loss`
//! - **Hinge Loss**: `hinge_loss`
//! - **Focal Loss**: `focal_loss`
//! - **Cosine Embedding Loss**: `cosine_embedding_loss`

use crate::tensor::Tensor;

// ============================================================================
// MSE Loss (Mean Squared Error)
// ============================================================================

/// Вычисляет символьный граф для Среднеквадратичной Ошибки (MSE) - сумма.
///
/// Формула: `MSE = sum((y_pred - y_true)^2)`.
///
/// # Аргументы
///
/// * `y_pred` - Символьный `Tensor` с предсказаниями модели.
/// * `y_true` - Символьный `Tensor` с истинными значениями.
///
/// # Возвращает
///
/// Новый символьный `Tensor`, представляющий скалярный узел (loss) в графе.
pub fn mse_loss(y_pred: &Tensor, y_true: &Tensor) -> Tensor {
    let error = y_pred - y_true;
    let squared_error = &error * &error;
    squared_error.sum()
}

/// Вычисляет MSE со средним значением вместо суммы.
///
/// Формула: `MSE = mean((y_pred - y_true)^2)`.
pub fn mse_loss_mean(y_pred: &Tensor, y_true: &Tensor) -> Tensor {
    let error = y_pred - y_true;
    let squared_error = &error * &error;
    squared_error.mean()
}

// ============================================================================
// L1 Loss (Mean Absolute Error)
// ============================================================================

/// Вычисляет L1 Loss (сумма абсолютных ошибок).
///
/// Формула: `L1 = sum(|y_pred - y_true|)`.
pub fn l1_loss(y_pred: &Tensor, y_true: &Tensor) -> Tensor {
    let error = y_pred - y_true;
    error.abs().sum()
}

/// Вычисляет L1 Loss со средним значением.
///
/// Формула: `MAE = mean(|y_pred - y_true|)`.
pub fn l1_loss_mean(y_pred: &Tensor, y_true: &Tensor) -> Tensor {
    let error = y_pred - y_true;
    error.abs().mean()
}

// ============================================================================
// Smooth L1 Loss (Huber Loss)
// ============================================================================

/// Вычисляет Smooth L1 Loss (Huber Loss).
///
/// Huber Loss менее чувствителен к выбросам, чем MSE.
/// Использует квадратичную функцию для малых ошибок и линейную для больших.
///
/// Формула:
/// ```text
/// loss = 0.5 * x^2           if |x| < beta
/// loss = beta * (|x| - 0.5 * beta)  otherwise
/// ```
///
/// # Аргументы
///
/// * `y_pred` - Предсказания модели
/// * `y_true` - Истинные значения
/// * `beta` - Порог перехода от квадратичной к линейной части (default: 1.0)
pub fn smooth_l1_loss(y_pred: &Tensor, y_true: &Tensor, beta: f32) -> Tensor {
    let error = y_pred - y_true;
    let abs_error = error.abs();

    // Создаем литерал для beta
    let beta_tensor = Tensor::scalar(&y_pred.context, beta);
    let half = Tensor::scalar(&y_pred.context, 0.5);

    // mask = |error| < beta (returns 1.0 where true, 0.0 where false)
    // Используем: quadratic_part = 0.5 * error^2 / beta
    //            linear_part = |error| - 0.5 * beta
    // Для упрощения используем приближение через clamp

    // Quadratic part: 0.5 * min(|error|, beta)^2 / beta
    let clamped = abs_error.clamp(0.0, beta);
    let quadratic = &(&clamped * &clamped) * &(&half / &beta_tensor);

    // Linear part: max(|error| - beta, 0)
    // linear_excess = max(|error| - beta, 0) = clamp(|error| - beta, 0, inf)
    let excess = &abs_error - &beta_tensor;
    let linear_excess = excess.clamp(0.0, f32::MAX);

    // Total: quadratic + linear_excess
    let total = &quadratic + &linear_excess;
    total.sum()
}

/// Вычисляет Huber Loss со средним значением.
pub fn huber_loss(y_pred: &Tensor, y_true: &Tensor, delta: f32) -> Tensor {
    smooth_l1_loss(y_pred, y_true, delta)
}

// ============================================================================
// Cross-Entropy Loss
// ============================================================================

/// Вычисляет Cross-Entropy Loss для задач классификации.
///
/// Формула: `CE = -sum(y_true * log(y_pred + eps))`
///
/// # Аргументы
///
/// * `y_pred` - Предсказанные вероятности (после softmax)
/// * `y_true` - One-hot encoded истинные метки
/// * `eps` - Малое значение для численной стабильности
pub fn cross_entropy_loss(y_pred: &Tensor, y_true: &Tensor, eps: f32) -> Tensor {
    let eps_tensor = Tensor::scalar(&y_pred.context, eps);
    let stabilized = y_pred + &eps_tensor;
    let log_pred = stabilized.log();
    let ce = y_true * &log_pred;
    ce.sum().neg()
}

/// Cross-Entropy Loss с Label Smoothing.
///
/// Label smoothing помогает предотвратить overconfident предсказания.
///
/// # Аргументы
///
/// * `y_pred` - Предсказанные вероятности
/// * `y_true` - One-hot encoded истинные метки
/// * `smoothing` - Коэффициент сглаживания (0.0 - 1.0, обычно 0.1)
/// * `num_classes` - Количество классов
/// * `eps` - Малое значение для численной стабильности
pub fn cross_entropy_with_label_smoothing(
    y_pred: &Tensor,
    y_true: &Tensor,
    smoothing: f32,
    num_classes: usize,
    eps: f32,
) -> Tensor {
    // smoothed_labels = y_true * (1 - smoothing) + smoothing / num_classes
    let one_minus_smooth = Tensor::scalar(&y_pred.context, 1.0 - smoothing);
    let smooth_value = Tensor::scalar(&y_pred.context, smoothing / num_classes as f32);

    let smoothed = &(y_true * &one_minus_smooth) + &smooth_value;

    let eps_tensor = Tensor::scalar(&y_pred.context, eps);
    let stabilized = y_pred + &eps_tensor;
    let log_pred = stabilized.log();
    let ce = &smoothed * &log_pred;
    ce.sum().neg()
}

// ============================================================================
// Binary Cross-Entropy Loss
// ============================================================================

/// Вычисляет Binary Cross-Entropy Loss.
///
/// Формула: `BCE = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))`
///
/// # Аргументы
///
/// * `y_pred` - Предсказанные вероятности (0-1, после sigmoid)
/// * `y_true` - Истинные метки (0 или 1)
/// * `eps` - Малое значение для численной стабильности
pub fn binary_cross_entropy(y_pred: &Tensor, y_true: &Tensor, eps: f32) -> Tensor {
    let eps_tensor = Tensor::scalar(&y_pred.context, eps);
    let one = Tensor::scalar(&y_pred.context, 1.0);

    // log(y_pred + eps)
    let log_pred = (&(y_pred + &eps_tensor)).log();

    // log(1 - y_pred + eps)
    let one_minus_pred = &one - y_pred;
    let log_one_minus_pred = (&one_minus_pred + &eps_tensor).log();

    // y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)
    let one_minus_true = &one - y_true;
    let term1 = y_true * &log_pred;
    let term2 = &one_minus_true * &log_one_minus_pred;

    let bce = &term1 + &term2;
    bce.sum().neg()
}

/// Вычисляет BCE с логитами (численно стабильная версия).
///
/// Использует logits напрямую без предварительного применения sigmoid.
/// Формула: `BCE = max(x, 0) - x * y + log(1 + exp(-|x|))`
///
/// # Аргументы
///
/// * `logits` - Сырые логиты (до sigmoid)
/// * `y_true` - Истинные метки (0 или 1)
pub fn bce_with_logits(logits: &Tensor, y_true: &Tensor) -> Tensor {
    // Численно стабильная формула:
    // max(x, 0) - x * y + log(1 + exp(-|x|))

    let one = Tensor::scalar(&logits.context, 1.0);

    // max(x, 0) = clamp(x, 0, inf)
    let max_x_0 = logits.clamp(0.0, f32::MAX);

    // x * y
    let x_times_y = logits * y_true;

    // |x|
    let abs_x = logits.abs();

    // exp(-|x|)
    let neg_abs_x = abs_x.neg();
    let exp_neg_abs = neg_abs_x.exp();

    // 1 + exp(-|x|)
    let one_plus_exp = &one + &exp_neg_abs;

    // log(1 + exp(-|x|))
    let log_term = one_plus_exp.log();

    // max(x, 0) - x * y + log(1 + exp(-|x|))
    let loss = &(&max_x_0 - &x_times_y) + &log_term;
    loss.sum()
}

// ============================================================================
// KL Divergence
// ============================================================================

/// Вычисляет KL Divergence (Kullback-Leibler Divergence).
///
/// Формула: `KL(P || Q) = sum(P * log(P / Q))`
///
/// # Аргументы
///
/// * `p` - Истинное распределение
/// * `q` - Предсказанное распределение
/// * `eps` - Малое значение для численной стабильности
pub fn kl_divergence(p: &Tensor, q: &Tensor, eps: f32) -> Tensor {
    let eps_tensor = Tensor::scalar(&p.context, eps);

    // log(P / Q) = log(P + eps) - log(Q + eps)
    let log_p = (&(p + &eps_tensor)).log();
    let log_q = (&(q + &eps_tensor)).log();
    let log_ratio = &log_p - &log_q;

    // P * log(P / Q)
    let kl = p * &log_ratio;
    kl.sum()
}

// ============================================================================
// Negative Log Likelihood Loss
// ============================================================================

/// Вычисляет Negative Log Likelihood Loss.
///
/// Используется с log-softmax выходами.
///
/// Формула: `NLL = -sum(y_true * log_probs)`
///
/// # Аргументы
///
/// * `log_probs` - Log-вероятности (выход log_softmax)
/// * `y_true` - One-hot encoded истинные метки
pub fn nll_loss(log_probs: &Tensor, y_true: &Tensor) -> Tensor {
    let nll = y_true * log_probs;
    nll.sum().neg()
}

// ============================================================================
// Hinge Loss
// ============================================================================

/// Вычисляет Hinge Loss для SVM-подобных классификаторов.
///
/// Формула: `Hinge = sum(max(0, margin - y_true * y_pred))`
///
/// # Аргументы
///
/// * `y_pred` - Предсказания модели
/// * `y_true` - Истинные метки (-1 или +1)
/// * `margin` - Маржа (default: 1.0)
pub fn hinge_loss(y_pred: &Tensor, y_true: &Tensor, margin: f32) -> Tensor {
    let margin_tensor = Tensor::scalar(&y_pred.context, margin);

    // margin - y_true * y_pred
    let prod = y_true * y_pred;
    let diff = &margin_tensor - &prod;

    // max(0, diff)
    let hinge = diff.clamp(0.0, f32::MAX);
    hinge.sum()
}

/// Вычисляет Squared Hinge Loss.
///
/// Формула: `SquaredHinge = sum(max(0, margin - y_true * y_pred)^2)`
pub fn squared_hinge_loss(y_pred: &Tensor, y_true: &Tensor, margin: f32) -> Tensor {
    let margin_tensor = Tensor::scalar(&y_pred.context, margin);

    let prod = y_true * y_pred;
    let diff = &margin_tensor - &prod;
    let hinge = diff.clamp(0.0, f32::MAX);
    let squared = &hinge * &hinge;
    squared.sum()
}

// ============================================================================
// Focal Loss
// ============================================================================

/// Вычисляет Focal Loss для задач с несбалансированными классами.
///
/// Focal Loss уменьшает вклад хорошо классифицированных примеров,
/// позволяя модели сосредоточиться на сложных примерах.
///
/// Формула: `FL = -alpha * (1 - p_t)^gamma * log(p_t)`
///
/// где `p_t = p если y=1, иначе 1-p`
///
/// # Аргументы
///
/// * `y_pred` - Предсказанные вероятности
/// * `y_true` - One-hot encoded истинные метки
/// * `alpha` - Балансирующий коэффициент (default: 0.25)
/// * `gamma` - Фокусирующий параметр (default: 2.0)
/// * `eps` - Малое значение для численной стабильности
pub fn focal_loss(
    y_pred: &Tensor,
    y_true: &Tensor,
    alpha: f32,
    gamma: f32,
    eps: f32,
) -> Tensor {
    let alpha_tensor = Tensor::scalar(&y_pred.context, alpha);
    let one = Tensor::scalar(&y_pred.context, 1.0);
    let eps_tensor = Tensor::scalar(&y_pred.context, eps);

    // p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    let one_minus_true = &one - y_true;
    let one_minus_pred = &one - y_pred;
    let p_t = &(y_true * y_pred) + &(&one_minus_true * &one_minus_pred);

    // (1 - p_t)^gamma - используем exp(gamma * log(1 - p_t))
    let one_minus_pt = &one - &p_t;
    let one_minus_pt_stable = &one_minus_pt + &eps_tensor;
    let log_one_minus_pt = one_minus_pt_stable.log();
    let gamma_tensor = Tensor::scalar(&y_pred.context, gamma);
    let gamma_log = &gamma_tensor * &log_one_minus_pt;
    let focal_weight = gamma_log.exp();

    // log(p_t + eps)
    let p_t_stable = &p_t + &eps_tensor;
    let log_pt = p_t_stable.log();

    // -alpha * focal_weight * log(p_t)
    let focal = &(&alpha_tensor * &focal_weight) * &log_pt;
    focal.sum().neg()
}

// ============================================================================
// Cosine Embedding Loss
// ============================================================================

/// Вычисляет Cosine Embedding Loss.
///
/// Используется для обучения embeddings, где похожие элементы
/// должны иметь высокое косинусное сходство.
///
/// Формула:
/// ```text
/// loss = 1 - cos(x1, x2)      if y = 1
/// loss = max(0, cos(x1, x2) - margin)  if y = -1
/// ```
///
/// # Аргументы
///
/// * `x1` - Первый вектор
/// * `x2` - Второй вектор
/// * `y` - Метка (1 для похожих, -1 для непохожих)
/// * `margin` - Маржа для негативных пар (default: 0.0)
/// * `eps` - Малое значение для численной стабильности
pub fn cosine_embedding_loss(
    x1: &Tensor,
    x2: &Tensor,
    y: &Tensor, // 1 или -1 для каждой пары
    margin: f32,
    eps: f32,
) -> Tensor {
    let one = Tensor::scalar(&x1.context, 1.0);
    let margin_tensor = Tensor::scalar(&x1.context, margin);
    let eps_tensor = Tensor::scalar(&x1.context, eps);

    // Cosine similarity: cos = (x1 · x2) / (||x1|| * ||x2||)
    // Для упрощения используем dot product и нормы
    let dot = &(x1 * x2);
    let dot_sum = dot.sum();

    // ||x1||^2 и ||x2||^2
    let x1_sq = &(x1 * x1);
    let x2_sq = &(x2 * x2);
    let norm1_sq = x1_sq.sum();
    let norm2_sq = x2_sq.sum();

    // ||x1|| * ||x2|| + eps
    let norm1 = norm1_sq.sqrt();
    let norm2 = norm2_sq.sqrt();
    let norm_prod = &(&norm1 * &norm2) + &eps_tensor;

    // cos = dot / norm_prod
    let cos_sim = &dot_sum / &norm_prod;

    // Для y = 1: loss = 1 - cos
    // Для y = -1: loss = max(0, cos - margin)
    // Комбинируем: loss = (1 + y) / 2 * (1 - cos) + (1 - y) / 2 * max(0, cos - margin)

    let half = Tensor::scalar(&x1.context, 0.5);
    let one_plus_y = &one + y;
    let one_minus_y = &one - y;

    // Positive pair loss: (1 - cos)
    let pos_loss = &one - &cos_sim;

    // Negative pair loss: max(0, cos - margin)
    let neg_diff = &cos_sim - &margin_tensor;
    let neg_loss = neg_diff.clamp(0.0, f32::MAX);

    // Combined loss
    let pos_weight = &one_plus_y * &half;
    let neg_weight = &one_minus_y * &half;

    let weighted_pos = &pos_weight * &pos_loss;
    let weighted_neg = &neg_weight * &neg_loss;

    &weighted_pos + &weighted_neg
}

// ============================================================================
// Triplet Margin Loss
// ============================================================================

/// Вычисляет Triplet Margin Loss для metric learning.
///
/// Формула: `loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)`
///
/// # Аргументы
///
/// * `anchor` - Якорный вектор
/// * `positive` - Позитивный вектор (должен быть близок к anchor)
/// * `negative` - Негативный вектор (должен быть далек от anchor)
/// * `margin` - Маржа между позитивной и негативной дистанцией
pub fn triplet_margin_loss(
    anchor: &Tensor,
    positive: &Tensor,
    negative: &Tensor,
    margin: f32,
) -> Tensor {
    let margin_tensor = Tensor::scalar(&anchor.context, margin);

    // d(anchor, positive) = ||anchor - positive||^2
    let diff_pos = anchor - positive;
    let dist_pos = (&diff_pos * &diff_pos).sum();

    // d(anchor, negative) = ||anchor - negative||^2
    let diff_neg = anchor - negative;
    let dist_neg = (&diff_neg * &diff_neg).sum();

    // loss = max(0, dist_pos - dist_neg + margin)
    let diff = &(&dist_pos - &dist_neg) + &margin_tensor;
    diff.clamp(0.0, f32::MAX)
}

// ============================================================================
// Margin Ranking Loss
// ============================================================================

/// Вычисляет Margin Ranking Loss.
///
/// Формула: `loss = max(0, -y * (x1 - x2) + margin)`
///
/// # Аргументы
///
/// * `x1` - Первый вход
/// * `x2` - Второй вход
/// * `y` - Метка: 1 если x1 должен быть больше x2, -1 иначе
/// * `margin` - Маржа
pub fn margin_ranking_loss(x1: &Tensor, x2: &Tensor, y: &Tensor, margin: f32) -> Tensor {
    let margin_tensor = Tensor::scalar(&x1.context, margin);

    // -y * (x1 - x2) + margin
    let diff = x1 - x2;
    let neg_y = y.neg();
    let scaled = &neg_y * &diff;
    let with_margin = &scaled + &margin_tensor;

    // max(0, ...)
    with_margin.clamp(0.0, f32::MAX).sum()
}
