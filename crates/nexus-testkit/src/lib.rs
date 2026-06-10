//! # nexus-testkit
//!
//! The CI parity layer. Reference outputs are exported **once** from the
//! upstream Python/onconpc pipeline to committed Parquet files under
//! `tests/snapshots/`; every CI run then compares the Rust pipeline's output
//! against those snapshots using Polars — no Python at test time.
//!
//! This crate provides:
//! - [`assert_matrix_near`] / [`compare_matrices`]: tolerant feature-matrix diff.
//! - Property helpers ([`assert_probabilities_valid`], [`assert_sbs96_width`],
//!   [`assert_shap_local_accuracy`]) for invariants that need no snapshot.

use nexus_core::{error::Result, FeatureMatrix, NexusError};

/// Absolute + relative tolerance for numeric comparisons.
#[derive(Debug, Clone, Copy)]
pub struct Tolerance {
    pub rtol: f64,
    pub atol: f64,
}

impl Tolerance {
    /// Default for probability outputs (tight).
    pub const PROBABILITY: Tolerance = Tolerance {
        rtol: 1e-4,
        atol: 1e-6,
    };
    /// Default for SHAP values (slightly looser).
    pub const SHAP: Tolerance = Tolerance {
        rtol: 1e-3,
        atol: 1e-5,
    };

    fn close(&self, a: f64, b: f64) -> bool {
        (a - b).abs() <= self.atol + self.rtol * b.abs()
    }
}

/// Compare two feature matrices element-wise within tolerance.
///
/// Column/row identity (names, order, shape) must match exactly; only the
/// numeric values are compared with tolerance.
pub fn compare_matrices(
    actual: &FeatureMatrix,
    expected: &FeatureMatrix,
    tol: Tolerance,
) -> Result<()> {
    if actual.feature_names != expected.feature_names {
        return Err(NexusError::invariant(format!(
            "feature name/order mismatch:\n  actual:   {:?}\n  expected: {:?}",
            actual.feature_names, expected.feature_names
        )));
    }
    if actual.sample_ids != expected.sample_ids {
        return Err(NexusError::invariant("sample id/order mismatch"));
    }
    if actual.values.dim() != expected.values.dim() {
        return Err(NexusError::ShapeMismatch(format!(
            "{:?} vs {:?}",
            actual.values.dim(),
            expected.values.dim()
        )));
    }
    for ((i, j), &a) in actual.values.indexed_iter() {
        let e = expected.values[[i, j]];
        if !tol.close(a, e) {
            return Err(NexusError::invariant(format!(
                "value mismatch at sample '{}', feature '{}': actual {a}, expected {e}",
                actual.sample_ids[i], actual.feature_names[j]
            )));
        }
    }
    Ok(())
}

/// Compare a computed matrix against a committed Parquet snapshot.
pub fn assert_matrix_near(
    actual: &FeatureMatrix,
    snapshot_parquet: impl AsRef<std::path::Path>,
    tol: Tolerance,
) -> Result<()> {
    let expected = nexus_data::read_parquet(snapshot_parquet)?;
    compare_matrices(actual, &expected, tol)
}

/// Property: every probability row is non-negative and sums to ~1.
pub fn assert_probabilities_valid(rows: &[Vec<f64>]) -> Result<()> {
    for (i, row) in rows.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        if (sum - 1.0).abs() > 1e-6 {
            return Err(NexusError::invariant(format!(
                "probability row {i} sums to {sum}, expected 1.0"
            )));
        }
        if row.iter().any(|&p| !(-1e-9..=1.0 + 1e-9).contains(&p)) {
            return Err(NexusError::invariant(format!(
                "probability row {i} has out-of-range entries"
            )));
        }
    }
    Ok(())
}

/// Property: an SBS96 matrix has exactly 96 columns.
pub fn assert_sbs96_width(channels: &[String]) -> Result<()> {
    if channels.len() != 96 {
        return Err(NexusError::invariant(format!(
            "expected 96 SBS channels, found {}",
            channels.len()
        )));
    }
    Ok(())
}

/// Property: SHAP values satisfy local accuracy: `sum(phi_c) + base_c == margin_c`.
pub fn assert_shap_local_accuracy(
    phi: &[f64],
    base: f64,
    margin: f64,
    tol: Tolerance,
) -> Result<()> {
    let recon: f64 = phi.iter().sum::<f64>() + base;
    if !tol.close(recon, margin) {
        return Err(NexusError::invariant(format!(
            "SHAP local accuracy violated: sum(phi)+base = {recon}, margin = {margin}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn m(values: ndarray::Array2<f64>) -> FeatureMatrix {
        FeatureMatrix::new(
            vec!["s1".into(), "s2".into()],
            vec!["A".into(), "B".into()],
            values,
        )
        .unwrap()
    }

    #[test]
    fn matrices_within_tolerance_pass() {
        let a = m(array![[1.0, 2.0], [3.0, 4.0]]);
        let b = m(array![[1.0 + 1e-7, 2.0], [3.0, 4.0 - 1e-7]]);
        assert!(compare_matrices(&a, &b, Tolerance::PROBABILITY).is_ok());
    }

    #[test]
    fn matrices_out_of_tolerance_fail() {
        let a = m(array![[1.0, 2.0], [3.0, 4.0]]);
        let b = m(array![[1.5, 2.0], [3.0, 4.0]]);
        assert!(compare_matrices(&a, &b, Tolerance::PROBABILITY).is_err());
    }

    #[test]
    fn snapshot_roundtrip_through_parquet() {
        let a = m(array![[1.0, 2.0], [3.0, 4.0]]);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("snap.parquet");
        nexus_data::write_parquet(&a, &path).unwrap();
        assert!(assert_matrix_near(&a, &path, Tolerance::PROBABILITY).is_ok());
    }

    #[test]
    fn probability_validation() {
        assert!(assert_probabilities_valid(&[vec![0.7, 0.3]]).is_ok());
        assert!(assert_probabilities_valid(&[vec![0.7, 0.5]]).is_err());
    }
}
