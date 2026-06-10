//! Exact polynomial-time Tree SHAP.
//!
//! This is a faithful Rust port of the algorithm from Lundberg et al.,
//! "Consistent Individualized Feature Attribution for Tree Ensembles"
//! (the `TreeExplainer` used by the Python `shap` package). It computes, for a
//! single sample, the exact Shapley value of every feature with respect to a
//! tree's output, in O(L * D^2) time per tree (L leaves, D depth).
//!
//! Correctness is anchored by the SHAP *local accuracy* (efficiency) property:
//! `sum_i phi_i + base_value == model_margin(x)` for every class. This is
//! asserted in the unit tests and is the same invariant `nexus-testkit` uses
//! as a property test in CI.

use ndarray::ArrayView1;

use nexus_ml::tree::{Tree, TreeEnsemble};

/// One element of the "unique path" of features encountered from the root.
#[derive(Clone, Copy, Default)]
struct PathElement {
    feature_index: i64,
    zero_fraction: f64,
    one_fraction: f64,
    pweight: f64,
}

/// Grow the subset path by one feature (Lundberg et al. EXTEND).
fn extend_path(
    path: &mut [PathElement],
    unique_depth: usize,
    zero_fraction: f64,
    one_fraction: f64,
    feature_index: i64,
) {
    path[unique_depth] = PathElement {
        feature_index,
        zero_fraction,
        one_fraction,
        pweight: if unique_depth == 0 { 1.0 } else { 0.0 },
    };
    if unique_depth == 0 {
        return;
    }
    let ud = unique_depth as f64;
    for i in (0..unique_depth).rev() {
        let fi = i as f64;
        path[i + 1].pweight += one_fraction * path[i].pweight * (fi + 1.0) / (ud + 1.0);
        path[i].pweight = zero_fraction * path[i].pweight * (ud - fi) / (ud + 1.0);
    }
}

/// Sum of path weights that would result from unwinding `path_index`,
/// without mutating the path (Lundberg et al. UNWOUND-PATH-SUM).
fn unwound_path_sum(path: &[PathElement], unique_depth: usize, path_index: usize) -> f64 {
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let mut next_one_portion = path[unique_depth].pweight;
    let mut total = 0.0;
    let ud = unique_depth as f64;
    for i in (0..unique_depth).rev() {
        let fi = i as f64;
        if one_fraction != 0.0 {
            let tmp = next_one_portion * (ud + 1.0) / ((fi + 1.0) * one_fraction);
            total += tmp;
            next_one_portion = path[i].pweight - tmp * zero_fraction * (ud - fi) / (ud + 1.0);
        } else if zero_fraction != 0.0 {
            total += (path[i].pweight / zero_fraction) / ((ud - fi) / (ud + 1.0));
        }
    }
    total
}

/// Undo a previous EXTEND for `path_index` (Lundberg et al. UNWIND).
fn unwind_path(path: &mut [PathElement], unique_depth: usize, path_index: usize) {
    let one_fraction = path[path_index].one_fraction;
    let zero_fraction = path[path_index].zero_fraction;
    let mut next_one_portion = path[unique_depth].pweight;
    let ud = unique_depth as f64;
    for i in (0..unique_depth).rev() {
        let fi = i as f64;
        if one_fraction != 0.0 {
            let tmp = path[i].pweight;
            path[i].pweight = next_one_portion * (ud + 1.0) / ((fi + 1.0) * one_fraction);
            next_one_portion = tmp - path[i].pweight * zero_fraction * (ud - fi) / (ud + 1.0);
        } else {
            path[i].pweight = (path[i].pweight * (ud + 1.0)) / (zero_fraction * (ud - fi));
        }
    }
    for i in path_index..unique_depth {
        path[i].feature_index = path[i + 1].feature_index;
        path[i].zero_fraction = path[i + 1].zero_fraction;
        path[i].one_fraction = path[i + 1].one_fraction;
    }
}

#[allow(clippy::too_many_arguments)]
fn recurse(
    tree: &Tree,
    x: &ArrayView1<f64>,
    phi: &mut [f64],
    node: usize,
    unique_depth: usize,
    parent_path: &[PathElement],
    parent_zero_fraction: f64,
    parent_one_fraction: f64,
    parent_feature_index: i64,
) {
    // Own a working copy of the parent's path, with room for one more element.
    let mut path = Vec::with_capacity(unique_depth + 2);
    path.extend_from_slice(&parent_path[..=unique_depth]);
    path.push(PathElement::default());

    extend_path(
        &mut path,
        unique_depth,
        parent_zero_fraction,
        parent_one_fraction,
        parent_feature_index,
    );

    if tree.is_leaf(node) {
        let leaf_value = tree.split_condition[node];
        for i in 1..=unique_depth {
            let w = unwound_path_sum(&path, unique_depth, i);
            let el = path[i];
            phi[el.feature_index as usize] += w * (el.one_fraction - el.zero_fraction) * leaf_value;
        }
        return;
    }

    let split_feature = tree.split_index[node] as i64;
    let cleft = tree.left[node] as usize;
    let cright = tree.right[node] as usize;
    let x_val = x[split_feature as usize];
    let go_left = if x_val.is_nan() {
        tree.default_left[node]
    } else {
        x_val < tree.split_condition[node]
    };
    let (hot, cold) = if go_left {
        (cleft, cright)
    } else {
        (cright, cleft)
    };

    let w = tree.cover[node];
    let hot_zero_fraction = if w != 0.0 { tree.cover[hot] / w } else { 0.0 };
    let cold_zero_fraction = if w != 0.0 { tree.cover[cold] / w } else { 0.0 };

    let mut incoming_zero_fraction = 1.0;
    let mut incoming_one_fraction = 1.0;
    let mut unique_depth = unique_depth;

    // Has this feature already been split on along the path?
    let mut path_index = 0;
    while path_index <= unique_depth {
        if path[path_index].feature_index == split_feature {
            break;
        }
        path_index += 1;
    }
    if path_index != unique_depth + 1 {
        incoming_zero_fraction = path[path_index].zero_fraction;
        incoming_one_fraction = path[path_index].one_fraction;
        unwind_path(&mut path, unique_depth, path_index);
        unique_depth -= 1;
    }

    recurse(
        tree,
        x,
        phi,
        hot,
        unique_depth + 1,
        &path,
        hot_zero_fraction * incoming_zero_fraction,
        incoming_one_fraction,
        split_feature,
    );
    recurse(
        tree,
        x,
        phi,
        cold,
        unique_depth + 1,
        &path,
        cold_zero_fraction * incoming_zero_fraction,
        0.0,
        split_feature,
    );
}

/// Per-feature SHAP values of a single tree for one sample.
pub fn tree_shap(tree: &Tree, x: &ArrayView1<f64>, n_features: usize) -> Vec<f64> {
    let mut phi = vec![0.0; n_features];
    // Root invocation: depth 0, incoming fractions 1, no parent feature.
    let seed = vec![PathElement::default()];
    recurse(tree, x, &mut phi, 0, 0, &seed, 1.0, 1.0, -1);
    phi
}

/// Cover-weighted mean leaf value of a tree: its SHAP base (expected) value.
pub fn tree_expected_value(tree: &Tree) -> f64 {
    fn rec(tree: &Tree, node: usize) -> f64 {
        if tree.is_leaf(node) {
            return tree.split_condition[node];
        }
        let w = tree.cover[node];
        if w == 0.0 {
            return 0.0;
        }
        let l = tree.left[node] as usize;
        let r = tree.right[node] as usize;
        (tree.cover[l] * rec(tree, l) + tree.cover[r] * rec(tree, r)) / w
    }
    rec(tree, 0)
}

/// SHAP values for an ensemble, returned as `[n_class][n_features]`, plus the
/// per-class base values.
pub struct EnsembleShap {
    pub values: Vec<Vec<f64>>,
    pub base_values: Vec<f64>,
}

/// Compute exact SHAP values for one sample across all classes.
pub fn ensemble_shap(ensemble: &TreeEnsemble, x: &ArrayView1<f64>) -> EnsembleShap {
    let n_class = ensemble.n_outputs();
    let n_feat = ensemble.num_feature.max(
        ensemble
            .trees
            .iter()
            .flat_map(|t| t.split_index.iter())
            .map(|&s| s as usize + 1)
            .max()
            .unwrap_or(0),
    );
    let mut values = vec![vec![0.0; n_feat]; n_class];
    let mut base_values = vec![0.0; n_class];
    for tree in &ensemble.trees {
        let phi = tree_shap(tree, x, n_feat);
        let c = tree.class_index;
        for (acc, v) in values[c].iter_mut().zip(phi.iter()) {
            *acc += v;
        }
        base_values[c] += tree_expected_value(tree);
    }
    EnsembleShap {
        values,
        base_values,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn toy_model_json() -> String {
        // Two classes, each a stump on feature 0 at threshold 0.5.
        serde_json::json!({
            "learner": {
                "feature_names": ["x0", "x1"],
                "learner_model_param": {"base_score":"5E-1","num_class":"2","num_feature":"2"},
                "gradient_booster": {"model": {
                    "tree_info": [0, 1],
                    "trees": [
                        {"left_children":[1,-1,-1],"right_children":[2,-1,-1],
                         "split_indices":[0,0,0],"split_conditions":[0.5,2.0,-2.0],
                         "default_left":[1,0,0],"sum_hessian":[10.0,4.0,6.0]},
                        {"left_children":[1,-1,-1],"right_children":[2,-1,-1],
                         "split_indices":[1,0,0],"split_conditions":[0.5,-3.0,3.0],
                         "default_left":[1,0,0],"sum_hessian":[10.0,5.0,5.0]}
                    ]
                }}
            }
        })
        .to_string()
    }

    #[test]
    fn local_accuracy_holds() {
        // The defining SHAP property: sum(phi_c) + base_c == margin_c(x).
        let ens = TreeEnsemble::from_json_slice(toy_model_json().as_bytes()).unwrap();
        for x in [array![0.0, 0.0], array![1.0, 1.0], array![0.0, 1.0]] {
            let margins = ens.predict_margins(x.view());
            let shap = ensemble_shap(&ens, &x.view());
            for (c, &margin) in margins.iter().enumerate() {
                let sum_phi: f64 = shap.values[c].iter().sum();
                let reconstructed = sum_phi + shap.base_values[c];
                assert!(
                    (reconstructed - margin).abs() < 1e-9,
                    "class {c}: phi_sum+base={reconstructed} margin={margin}"
                );
            }
        }
    }

    #[test]
    fn unused_feature_has_zero_attribution() {
        // Class 0's tree only splits on feature 0, so feature 1 must get 0.
        let ens = TreeEnsemble::from_json_slice(toy_model_json().as_bytes()).unwrap();
        let x = array![1.0, 1.0];
        let shap = ensemble_shap(&ens, &x.view());
        assert!(shap.values[0][1].abs() < 1e-12);
    }
}
