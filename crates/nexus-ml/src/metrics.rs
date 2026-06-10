//! Classification metrics mirroring sklearn's `classification_report`
//! (per-class precision/recall/F1/support, accuracy, macro & weighted avgs),
//! plus posterior-cutoff filtering used in the OncoNPC evaluation.

use serde::{Deserialize, Serialize};

/// Per-class metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassMetrics {
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub support: usize,
}

/// A full classification report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationReport {
    /// Keyed by class name.
    pub per_class: Vec<(String, ClassMetrics)>,
    pub accuracy: f64,
    pub macro_avg: ClassMetrics,
    pub weighted_avg: ClassMetrics,
}

fn safe_div(a: f64, b: f64) -> f64 {
    if b == 0.0 {
        0.0
    } else {
        a / b
    }
}

/// Compute a classification report from integer class labels.
///
/// `y_true` and `y_pred` are class indices; `class_names[i]` names class `i`.
pub fn classification_report(
    y_true: &[usize],
    y_pred: &[usize],
    class_names: &[String],
) -> ClassificationReport {
    let n_classes = class_names.len();
    let mut tp = vec![0usize; n_classes];
    let mut fp = vec![0usize; n_classes];
    let mut fn_ = vec![0usize; n_classes];
    let mut support = vec![0usize; n_classes];

    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        support[t] += 1;
        if t == p {
            tp[t] += 1;
        } else {
            fp[p] += 1;
            fn_[t] += 1;
        }
    }

    let total = y_true.len();
    let correct: usize = (0..n_classes).map(|c| tp[c]).sum();
    let accuracy = safe_div(correct as f64, total as f64);

    let mut per_class = Vec::with_capacity(n_classes);
    let (mut macro_p, mut macro_r, mut macro_f) = (0.0, 0.0, 0.0);
    let (mut wp, mut wr, mut wf) = (0.0, 0.0, 0.0);

    for c in 0..n_classes {
        let precision = safe_div(tp[c] as f64, (tp[c] + fp[c]) as f64);
        let recall = safe_div(tp[c] as f64, (tp[c] + fn_[c]) as f64);
        let f1 = safe_div(2.0 * precision * recall, precision + recall);
        macro_p += precision;
        macro_r += recall;
        macro_f += f1;
        let w = support[c] as f64;
        wp += precision * w;
        wr += recall * w;
        wf += f1 * w;
        per_class.push((
            class_names[c].clone(),
            ClassMetrics {
                precision,
                recall,
                f1,
                support: support[c],
            },
        ));
    }

    let nc = n_classes.max(1) as f64;
    let t = total.max(1) as f64;
    ClassificationReport {
        per_class,
        accuracy,
        macro_avg: ClassMetrics {
            precision: macro_p / nc,
            recall: macro_r / nc,
            f1: macro_f / nc,
            support: total,
        },
        weighted_avg: ClassMetrics {
            precision: wp / t,
            recall: wr / t,
            f1: wf / t,
            support: total,
        },
    }
}

/// Keep only samples whose max posterior exceeds `cutoff`; returns the
/// retained sample indices and their predicted class indices. Mirrors
/// `get_sample_indices_and_labels_based_on_cut_off`.
pub fn filter_by_posterior_cutoff(
    pred_probs: &[Vec<f64>],
    cutoff: f64,
) -> (Vec<usize>, Vec<usize>) {
    let mut idxs = Vec::new();
    let mut preds = Vec::new();
    for (i, probs) in pred_probs.iter().enumerate() {
        let (argmax, &maxp) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        if maxp > cutoff {
            idxs.push(i);
            preds.push(argmax);
        }
    }
    (idxs, preds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perfect_prediction_is_accuracy_one() {
        let names = vec!["A".to_string(), "B".to_string()];
        let r = classification_report(&[0, 1, 0, 1], &[0, 1, 0, 1], &names);
        assert!((r.accuracy - 1.0).abs() < 1e-12);
        assert!((r.macro_avg.f1 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cutoff_filters_low_confidence() {
        let probs = vec![vec![0.9, 0.1], vec![0.6, 0.4], vec![0.55, 0.45]];
        let (idx, preds) = filter_by_posterior_cutoff(&probs, 0.7);
        assert_eq!(idx, vec![0]);
        assert_eq!(preds, vec![0]);
    }
}
