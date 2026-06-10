//! Native XGBoost tree-ensemble representation and JSON loader.
//!
//! We parse the JSON produced by `Booster.save_model("model.json")`
//! (XGBoost >= 1.0) rather than linking the XGBoost C library. This keeps
//! inference dependency-free *and* gives `nexus-shap` direct access to the
//! tree structure (children, split conditions, node cover) it needs for
//! exact Tree SHAP.
//!
//! Inference parity note: the upstream Python obtains probabilities via
//! `predict(output_margin=True)` followed by an explicit softmax
//! (`get_xgboost_latest_cancer_type_preds`). Because softmax is invariant to
//! a constant added to every class, the scalar `base_score` cancels out for
//! `multi:softprob` and does not affect probabilities; we still retain it for
//! SHAP base-value reporting.

use ndarray::ArrayView1;
use serde::Deserialize;

use nexus_core::error::{NexusError, Result};

/// A single decision tree in CSR-like parallel-array form (mirrors the
/// XGBoost JSON layout for cheap, cache-friendly traversal).
#[derive(Debug, Clone)]
pub struct Tree {
    /// `left_children[i]` (`-1` if node `i` is a leaf).
    pub left: Vec<i32>,
    /// `right_children[i]` (`-1` if node `i` is a leaf).
    pub right: Vec<i32>,
    /// Feature index tested at internal node `i`.
    pub split_index: Vec<i32>,
    /// For internal nodes: the split threshold. For leaves: the leaf value.
    pub split_condition: Vec<f64>,
    /// Whether a missing value routes left at node `i`.
    pub default_left: Vec<bool>,
    /// Node cover (`sum_hessian`); used by Tree SHAP weighting.
    pub cover: Vec<f64>,
    /// Output class this tree contributes to (for multiclass).
    pub class_index: usize,
}

impl Tree {
    #[inline]
    pub fn is_leaf(&self, node: usize) -> bool {
        self.left[node] == -1
    }

    /// Evaluate the tree on a dense feature row, returning the leaf value.
    ///
    /// Decision rule matches XGBoost: `x < threshold` routes left, else right.
    /// (Inputs here are dense, so the `default_left` path is unused but kept
    /// for completeness / NaN handling.)
    pub fn predict(&self, row: ArrayView1<f64>) -> f64 {
        let mut node = 0usize;
        while !self.is_leaf(node) {
            let feat = self.split_index[node] as usize;
            let x = row[feat];
            let go_left = if x.is_nan() {
                self.default_left[node]
            } else {
                x < self.split_condition[node]
            };
            node = if go_left {
                self.left[node] as usize
            } else {
                self.right[node] as usize
            };
        }
        self.split_condition[node]
    }
}

/// A gradient-boosted forest plus the metadata needed to turn margins into
/// class probabilities.
#[derive(Debug, Clone)]
pub struct TreeEnsemble {
    pub trees: Vec<Tree>,
    pub num_class: usize,
    pub num_feature: usize,
    pub base_score: f64,
    pub feature_names: Option<Vec<String>>,
}

impl TreeEnsemble {
    /// Number of output classes (>= 1).
    pub fn n_outputs(&self) -> usize {
        self.num_class.max(1)
    }

    /// Raw per-class margins for one sample (sum of leaf values per class).
    pub fn predict_margins(&self, row: ArrayView1<f64>) -> Vec<f64> {
        let mut margins = vec![0.0; self.n_outputs()];
        for tree in &self.trees {
            margins[tree.class_index] += tree.predict(row);
        }
        margins
    }

    /// Class probabilities for one sample via softmax over margins
    /// (multiclass) or sigmoid (single output).
    pub fn predict_proba_row(&self, row: ArrayView1<f64>) -> Vec<f64> {
        let margins = self.predict_margins(row);
        if self.n_outputs() == 1 {
            let p = 1.0 / (1.0 + (-(margins[0] + self.base_score)).exp());
            vec![1.0 - p, p]
        } else {
            softmax(&margins)
        }
    }

    /// Parse an ensemble from XGBoost `save_model` JSON bytes.
    pub fn from_json_slice(bytes: &[u8]) -> Result<Self> {
        let raw: XgbModel = serde_json::from_slice(bytes)?;
        raw.into_ensemble()
    }

    /// Parse an ensemble from an XGBoost JSON file on disk.
    pub fn from_json_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        Self::from_json_slice(&bytes)
    }
}

/// Numerically stable softmax.
pub fn softmax(margins: &[f64]) -> Vec<f64> {
    let max = margins.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = margins.iter().map(|m| (m - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

// ---------------------------------------------------------------------------
// JSON deserialization of the XGBoost model format.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct XgbModel {
    learner: Learner,
}

#[derive(Deserialize)]
struct Learner {
    #[serde(default)]
    feature_names: Option<Vec<String>>,
    gradient_booster: GradientBooster,
    learner_model_param: LearnerModelParam,
}

#[derive(Deserialize)]
struct GradientBooster {
    model: GbtModel,
}

#[derive(Deserialize)]
struct GbtModel {
    trees: Vec<RawTree>,
    #[serde(default)]
    tree_info: Vec<i32>,
}

#[derive(Deserialize)]
struct LearnerModelParam {
    #[serde(default)]
    base_score: Option<String>,
    #[serde(default)]
    num_class: Option<String>,
    #[serde(default)]
    num_feature: Option<String>,
}

#[derive(Deserialize)]
struct RawTree {
    left_children: Vec<i32>,
    right_children: Vec<i32>,
    split_indices: Vec<i32>,
    split_conditions: Vec<f64>,
    #[serde(default)]
    default_left: Vec<i32>,
    #[serde(default)]
    sum_hessian: Vec<f64>,
}

fn parse_xgb_number(s: &Option<String>, default: f64) -> f64 {
    s.as_ref()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(default)
}

impl XgbModel {
    fn into_ensemble(self) -> Result<TreeEnsemble> {
        let num_class = parse_xgb_number(&self.learner.learner_model_param.num_class, 0.0) as usize;
        let num_feature =
            parse_xgb_number(&self.learner.learner_model_param.num_feature, 0.0) as usize;
        let base_score = parse_xgb_number(&self.learner.learner_model_param.base_score, 0.5);

        let gbt = self.learner.gradient_booster.model;
        let tree_info = gbt.tree_info;
        let n_outputs = num_class.max(1);

        let mut trees = Vec::with_capacity(gbt.trees.len());
        for (i, rt) in gbt.trees.into_iter().enumerate() {
            let n = rt.left_children.len();
            if rt.right_children.len() != n
                || rt.split_indices.len() != n
                || rt.split_conditions.len() != n
            {
                return Err(NexusError::InvalidModel(format!(
                    "tree {i}: inconsistent node array lengths"
                )));
            }
            let default_left = if rt.default_left.len() == n {
                rt.default_left.iter().map(|&d| d != 0).collect()
            } else {
                vec![true; n]
            };
            let cover = if rt.sum_hessian.len() == n {
                rt.sum_hessian
            } else {
                vec![1.0; n]
            };
            let class_index = tree_info.get(i).copied().unwrap_or(0) as usize % n_outputs;
            trees.push(Tree {
                left: rt.left_children,
                right: rt.right_children,
                split_index: rt.split_indices,
                split_condition: rt.split_conditions,
                default_left,
                cover,
                class_index,
            });
        }

        Ok(TreeEnsemble {
            trees,
            num_class,
            num_feature,
            base_score,
            feature_names: self.learner.feature_names,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// A tiny hand-written 2-class model: 2 trees (one per class), each a
    /// single split on feature 0 at threshold 0.5.
    fn toy_model_json() -> String {
        // class 0 tree: x0 < 0.5 -> leaf 1.0 else leaf -1.0
        // class 1 tree: x0 < 0.5 -> leaf -1.0 else leaf 1.0
        serde_json::json!({
            "learner": {
                "feature_names": ["x0"],
                "learner_model_param": {
                    "base_score": "5E-1",
                    "num_class": "2",
                    "num_feature": "1"
                },
                "gradient_booster": {
                    "model": {
                        "tree_info": [0, 1],
                        "trees": [
                            {
                                "left_children": [1, -1, -1],
                                "right_children": [2, -1, -1],
                                "split_indices": [0, 0, 0],
                                "split_conditions": [0.5, 1.0, -1.0],
                                "default_left": [1, 0, 0],
                                "sum_hessian": [10.0, 5.0, 5.0]
                            },
                            {
                                "left_children": [1, -1, -1],
                                "right_children": [2, -1, -1],
                                "split_indices": [0, 0, 0],
                                "split_conditions": [0.5, -1.0, 1.0],
                                "default_left": [1, 0, 0],
                                "sum_hessian": [10.0, 5.0, 5.0]
                            }
                        ]
                    }
                }
            }
        })
        .to_string()
    }

    #[test]
    fn loads_and_predicts() {
        let ens = TreeEnsemble::from_json_slice(toy_model_json().as_bytes()).unwrap();
        assert_eq!(ens.num_class, 2);
        assert_eq!(ens.trees.len(), 2);
        assert_eq!(ens.trees[0].class_index, 0);
        assert_eq!(ens.trees[1].class_index, 1);

        // x0 = 0.0 -> class 0 margin 1.0, class 1 margin -1.0 -> class 0 wins.
        let probs = ens.predict_proba_row(array![0.0].view());
        assert!(probs[0] > probs[1]);
        assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-9);

        // x0 = 1.0 -> class 1 wins.
        let probs = ens.predict_proba_row(array![1.0].view());
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[2.0, 1.0, 0.1]);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
