//! Low-frequency feature/sample filtering.
//!
//! Ports `filter_by_threshold` and `filter_out_low_freq_feats_and_samples`
//! from `codes/utils_training.py`. A sample is dropped only if it is
//! low-signal in *both* the mutation and CNA groups (intersection); a feature
//! is dropped if it is rare in *either* group (union).

use std::collections::BTreeSet;

use ndarray::Array2;
use nexus_core::{error::Result, FeatureGroup, FeatureMatrix};

/// Thresholds controlling the low-frequency filter.
#[derive(Debug, Clone, Copy)]
pub struct FilterThresholds {
    /// Minimum number of nonzero features (within a group) for a sample to
    /// be retained.
    pub per_sample: usize,
    /// Minimum number of nonzero samples for a feature to be retained.
    pub per_feature: usize,
}

impl Default for FilterThresholds {
    fn default() -> Self {
        // Matches the Python defaults: threshold_per_sample=3, _per_feature=50.
        Self {
            per_sample: 3,
            per_feature: 50,
        }
    }
}

/// Result of applying the low-frequency filter.
#[derive(Debug, Clone)]
pub struct FilterOutcome {
    pub matrix: FeatureMatrix,
    /// Indices (into the *original* matrix) of dropped samples.
    pub dropped_samples: Vec<usize>,
    /// Names of dropped features.
    pub dropped_features: Vec<String>,
}

/// Within a chosen set of columns, find features and samples below threshold.
fn low_in_group(
    matrix: &FeatureMatrix,
    group_cols: &[usize],
    thr: FilterThresholds,
) -> (BTreeSet<usize>, BTreeSet<usize>) {
    let n_samples = matrix.n_samples();

    // Per-feature nonzero counts.
    let mut features_to_exclude = BTreeSet::new();
    for &col in group_cols {
        let nonzero = (0..n_samples)
            .filter(|&r| matrix.values[[r, col]] != 0.0)
            .count();
        if nonzero < thr.per_feature {
            features_to_exclude.insert(col);
        }
    }

    // Per-sample nonzero counts within this group.
    let mut samples_to_exclude = BTreeSet::new();
    for r in 0..n_samples {
        let nonzero = group_cols
            .iter()
            .filter(|&&c| matrix.values[[r, c]] != 0.0)
            .count();
        if nonzero < thr.per_sample {
            samples_to_exclude.insert(r);
        }
    }

    (features_to_exclude, samples_to_exclude)
}

/// Apply the OncoNPC low-frequency filter to a feature matrix.
pub fn low_frequency_filter(
    matrix: &FeatureMatrix,
    thr: FilterThresholds,
) -> Result<FilterOutcome> {
    let mutation_cols: Vec<usize> = (0..matrix.n_features())
        .filter(|&c| FeatureGroup::classify(&matrix.feature_names[c]) == FeatureGroup::Mutation)
        .collect();
    let cna_cols: Vec<usize> = (0..matrix.n_features())
        .filter(|&c| FeatureGroup::classify(&matrix.feature_names[c]) == FeatureGroup::Cna)
        .collect();

    let (mut_feats, mut_samples) = low_in_group(matrix, &mutation_cols, thr);
    let (cna_feats, cna_samples) = low_in_group(matrix, &cna_cols, thr);

    // Samples excluded only if low in BOTH groups (intersection).
    let samples_to_drop: BTreeSet<usize> =
        mut_samples.intersection(&cna_samples).copied().collect();
    // Features excluded if rare in EITHER group (union).
    let feats_to_drop: BTreeSet<usize> = mut_feats.union(&cna_feats).copied().collect();

    let kept_rows: Vec<usize> = (0..matrix.n_samples())
        .filter(|r| !samples_to_drop.contains(r))
        .collect();
    let kept_cols: Vec<usize> = (0..matrix.n_features())
        .filter(|c| !feats_to_drop.contains(c))
        .collect();

    let mut values = Array2::<f64>::zeros((kept_rows.len(), kept_cols.len()));
    for (nr, &r) in kept_rows.iter().enumerate() {
        for (nc, &c) in kept_cols.iter().enumerate() {
            values[[nr, nc]] = matrix.values[[r, c]];
        }
    }
    let sample_ids = kept_rows
        .iter()
        .map(|&r| matrix.sample_ids[r].clone())
        .collect();
    let feature_names = kept_cols
        .iter()
        .map(|&c| matrix.feature_names[c].clone())
        .collect();

    let dropped_features = feats_to_drop
        .iter()
        .map(|&c| matrix.feature_names[c].clone())
        .collect();

    Ok(FilterOutcome {
        matrix: FeatureMatrix::new(sample_ids, feature_names, values)?,
        dropped_samples: samples_to_drop.into_iter().collect(),
        dropped_features,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn drops_rare_feature_and_low_signal_sample() {
        // 4 samples, features: GENEA (mutation), GENEB (mutation), "X CNA"
        let matrix = FeatureMatrix::new(
            vec!["s1".into(), "s2".into(), "s3".into(), "s4".into()],
            vec!["GENEA".into(), "GENEB".into(), "X CNA".into()],
            array![
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        )
        .unwrap();

        let thr = FilterThresholds {
            per_sample: 1,
            per_feature: 2,
        };
        let out = low_frequency_filter(&matrix, thr).unwrap();

        // GENEB appears in 1 sample (< 2) -> dropped. "X CNA" never nonzero -> dropped.
        assert!(out.dropped_features.contains(&"GENEB".to_string()));
        assert!(out.dropped_features.contains(&"X CNA".to_string()));
        // s4 is all-zero -> low in BOTH mutation and CNA groups -> dropped.
        assert_eq!(out.matrix.n_samples(), 3);
        assert_eq!(out.dropped_samples, vec![3]);
        assert_eq!(out.matrix.feature_names, vec!["GENEA".to_string()]);
    }
}
