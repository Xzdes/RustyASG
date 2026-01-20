// --- File: src/losses.rs ---

//! Module containing implementations of loss functions in graph paradigm.
//!
//! Loss functions here are regular Rust functions that take
//! symbolic `Tensor` descriptors and add nodes to the graph necessary
//! for computing the error value.
//!
//! # Available Loss Functions
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

/// Computes symbolic graph for Mean Squared Error (MSE) - sum.
///
/// Formula: `MSE = sum((y_pred - y_true)^2)`.
///
/// # Arguments
///
/// * `y_pred` - Symbolic `Tensor` with model predictions.
/// * `y_true` - Symbolic `Tensor` with true values.
///
/// # Returns
///
/// New symbolic `Tensor` representing a scalar node (loss) in the graph.
pub fn mse_loss(y_pred: &Tensor, y_true: &Tensor) -> Tensor {
    let error = y_pred - y_true;
    let squared_error = &error * &error;
    squared_error.sum()
}

/// Computes MSE with mean value instead of sum.
///
/// Formula: `MSE = mean((y_pred - y_true)^2)`.
pub fn mse_loss_mean(y_pred: &Tensor, y_true: &Tensor) -> Tensor {
    let error = y_pred - y_true;
    let squared_error = &error * &error;
    squared_error.mean()
}

// ============================================================================
// L1 Loss (Mean Absolute Error)
// ============================================================================

/// Computes L1 Loss (sum of absolute errors).
///
/// Formula: `L1 = sum(|y_pred - y_true|)`.
pub fn l1_loss(y_pred: &Tensor, y_true: &Tensor) -> Tensor {
    let error = y_pred - y_true;
    error.abs().sum()
}

/// Computes L1 Loss with mean value.
///
/// Formula: `MAE = mean(|y_pred - y_true|)`.
pub fn l1_loss_mean(y_pred: &Tensor, y_true: &Tensor) -> Tensor {
    let error = y_pred - y_true;
    error.abs().mean()
}

// ============================================================================
// Smooth L1 Loss (Huber Loss)
// ============================================================================

/// Computes Smooth L1 Loss (Huber Loss).
///
/// Huber Loss is less sensitive to outliers than MSE.
/// Uses quadratic function for small errors and linear for large ones.
///
/// Formula:
/// ```text
/// loss = 0.5 * x^2           if |x| < beta
/// loss = beta * (|x| - 0.5 * beta)  otherwise
/// ```
///
/// # Arguments
///
/// * `y_pred` - Model predictions
/// * `y_true` - True values
/// * `beta` - Threshold for transition from quadratic to linear part (default: 1.0)
pub fn smooth_l1_loss(y_pred: &Tensor, y_true: &Tensor, beta: f32) -> Tensor {
    let error = y_pred - y_true;
    let abs_error = error.abs();

    // Create literal for beta
    let beta_tensor = Tensor::scalar(&y_pred.context, beta);
    let half = Tensor::scalar(&y_pred.context, 0.5);

    // mask = |error| < beta (returns 1.0 where true, 0.0 where false)
    // Using: quadratic_part = 0.5 * error^2 / beta
    //        linear_part = |error| - 0.5 * beta
    // For simplicity using approximation via clamp

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

/// Computes Huber Loss with mean value.
pub fn huber_loss(y_pred: &Tensor, y_true: &Tensor, delta: f32) -> Tensor {
    smooth_l1_loss(y_pred, y_true, delta)
}

// ============================================================================
// Cross-Entropy Loss
// ============================================================================

/// Computes Cross-Entropy Loss for classification tasks.
///
/// Formula: `CE = -sum(y_true * log(y_pred + eps))`
///
/// # Arguments
///
/// * `y_pred` - Predicted probabilities (after softmax)
/// * `y_true` - One-hot encoded true labels
/// * `eps` - Small value for numerical stability
pub fn cross_entropy_loss(y_pred: &Tensor, y_true: &Tensor, eps: f32) -> Tensor {
    let eps_tensor = Tensor::scalar(&y_pred.context, eps);
    let stabilized = y_pred + &eps_tensor;
    let log_pred = stabilized.log();
    let ce = y_true * &log_pred;
    ce.sum().neg()
}

/// Cross-Entropy Loss with Label Smoothing.
///
/// Label smoothing helps prevent overconfident predictions.
///
/// # Arguments
///
/// * `y_pred` - Predicted probabilities
/// * `y_true` - One-hot encoded true labels
/// * `smoothing` - Smoothing coefficient (0.0 - 1.0, typically 0.1)
/// * `num_classes` - Number of classes
/// * `eps` - Small value for numerical stability
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

/// Computes Binary Cross-Entropy Loss.
///
/// Formula: `BCE = -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))`
///
/// # Arguments
///
/// * `y_pred` - Predicted probabilities (0-1, after sigmoid)
/// * `y_true` - True labels (0 or 1)
/// * `eps` - Small value for numerical stability
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

/// Computes BCE with logits (numerically stable version).
///
/// Uses logits directly without prior sigmoid application.
/// Formula: `BCE = max(x, 0) - x * y + log(1 + exp(-|x|))`
///
/// # Arguments
///
/// * `logits` - Raw logits (before sigmoid)
/// * `y_true` - True labels (0 or 1)
pub fn bce_with_logits(logits: &Tensor, y_true: &Tensor) -> Tensor {
    // Numerically stable formula:
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

/// Computes KL Divergence (Kullback-Leibler Divergence).
///
/// Formula: `KL(P || Q) = sum(P * log(P / Q))`
///
/// # Arguments
///
/// * `p` - True distribution
/// * `q` - Predicted distribution
/// * `eps` - Small value for numerical stability
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

/// Computes Negative Log Likelihood Loss.
///
/// Used with log-softmax outputs.
///
/// Formula: `NLL = -sum(y_true * log_probs)`
///
/// # Arguments
///
/// * `log_probs` - Log probabilities (log_softmax output)
/// * `y_true` - One-hot encoded true labels
pub fn nll_loss(log_probs: &Tensor, y_true: &Tensor) -> Tensor {
    let nll = y_true * log_probs;
    nll.sum().neg()
}

// ============================================================================
// Hinge Loss
// ============================================================================

/// Computes Hinge Loss for SVM-like classifiers.
///
/// Formula: `Hinge = sum(max(0, margin - y_true * y_pred))`
///
/// # Arguments
///
/// * `y_pred` - Model predictions
/// * `y_true` - True labels (-1 or +1)
/// * `margin` - Margin (default: 1.0)
pub fn hinge_loss(y_pred: &Tensor, y_true: &Tensor, margin: f32) -> Tensor {
    let margin_tensor = Tensor::scalar(&y_pred.context, margin);

    // margin - y_true * y_pred
    let prod = y_true * y_pred;
    let diff = &margin_tensor - &prod;

    // max(0, diff)
    let hinge = diff.clamp(0.0, f32::MAX);
    hinge.sum()
}

/// Computes Squared Hinge Loss.
///
/// Formula: `SquaredHinge = sum(max(0, margin - y_true * y_pred)^2)`
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

/// Computes Focal Loss for imbalanced class tasks.
///
/// Focal Loss reduces the contribution of well-classified examples,
/// allowing the model to focus on hard examples.
///
/// Formula: `FL = -alpha * (1 - p_t)^gamma * log(p_t)`
///
/// where `p_t = p if y=1, else 1-p`
///
/// # Arguments
///
/// * `y_pred` - Predicted probabilities
/// * `y_true` - One-hot encoded true labels
/// * `alpha` - Balancing coefficient (default: 0.25)
/// * `gamma` - Focusing parameter (default: 2.0)
/// * `eps` - Small value for numerical stability
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

    // (1 - p_t)^gamma - using exp(gamma * log(1 - p_t))
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

/// Computes Cosine Embedding Loss.
///
/// Used for training embeddings where similar elements
/// should have high cosine similarity.
///
/// Formula:
/// ```text
/// loss = 1 - cos(x1, x2)      if y = 1
/// loss = max(0, cos(x1, x2) - margin)  if y = -1
/// ```
///
/// # Arguments
///
/// * `x1` - First vector
/// * `x2` - Second vector
/// * `y` - Label (1 for similar, -1 for dissimilar)
/// * `margin` - Margin for negative pairs (default: 0.0)
/// * `eps` - Small value for numerical stability
pub fn cosine_embedding_loss(
    x1: &Tensor,
    x2: &Tensor,
    y: &Tensor, // 1 or -1 for each pair
    margin: f32,
    eps: f32,
) -> Tensor {
    let one = Tensor::scalar(&x1.context, 1.0);
    let margin_tensor = Tensor::scalar(&x1.context, margin);
    let eps_tensor = Tensor::scalar(&x1.context, eps);

    // Cosine similarity: cos = (x1 · x2) / (||x1|| * ||x2||)
    // For simplicity using dot product and norms
    let dot = &(x1 * x2);
    let dot_sum = dot.sum();

    // ||x1||^2 and ||x2||^2
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

    // For y = 1: loss = 1 - cos
    // For y = -1: loss = max(0, cos - margin)
    // Combined: loss = (1 + y) / 2 * (1 - cos) + (1 - y) / 2 * max(0, cos - margin)

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

/// Computes Triplet Margin Loss for metric learning.
///
/// Formula: `loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)`
///
/// # Arguments
///
/// * `anchor` - Anchor vector
/// * `positive` - Positive vector (should be close to anchor)
/// * `negative` - Negative vector (should be far from anchor)
/// * `margin` - Margin between positive and negative distance
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

/// Computes Margin Ranking Loss.
///
/// Formula: `loss = max(0, -y * (x1 - x2) + margin)`
///
/// # Arguments
///
/// * `x1` - First input
/// * `x2` - Second input
/// * `y` - Label: 1 if x1 should be greater than x2, -1 otherwise
/// * `margin` - Margin
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
