//! Feature taxonomy and the canonical [`FeatureMatrix`] contract type.
//!
//! Ports the grouping/coloring/standardization logic from the Python
//! reference (`codes/utils.py`: `partition_feature_names_by_group`,
//! `get_color`, `standardize_feat_names`).

use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::error::{NexusError, Result};

/// The four biological/clinical feature families used by OncoNPC.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FeatureGroup {
    /// Somatic mutation gene counts (e.g. `ERBB2`).
    Mutation,
    /// Copy-number alteration events (e.g. `ERBB2 CNA`).
    Cna,
    /// Mutation signature weights (e.g. `SBS1`).
    Signature,
    /// Clinical covariates: `Age`, `Sex`.
    Clinical,
}

impl FeatureGroup {
    /// Classify a (standardized) feature name into its group.
    ///
    /// Mirrors `partition_feature_names_by_group`: the order of checks
    /// matters — signature is detected first, then clinical, then CNA,
    /// with mutation as the fallback.
    pub fn classify(feature_name: &str) -> FeatureGroup {
        if feature_name.contains("SBS") {
            FeatureGroup::Signature
        } else if feature_name == "Age" || feature_name == "Sex" {
            FeatureGroup::Clinical
        } else if feature_name.contains("CNA") {
            FeatureGroup::Cna
        } else {
            FeatureGroup::Mutation
        }
    }

    /// The plotting color used in SHAP explanation charts, matching the
    /// Python `get_color` mapping.
    pub fn color(self) -> &'static str {
        match self {
            FeatureGroup::Mutation => "red",
            FeatureGroup::Cna => "green",
            FeatureGroup::Signature => "blue",
            FeatureGroup::Clinical => "grey",
        }
    }

    /// Human-readable legend label used in explanation plots.
    pub fn legend_label(self) -> &'static str {
        match self {
            FeatureGroup::Mutation => "Somatic Mut.",
            FeatureGroup::Cna => "CNA events",
            FeatureGroup::Signature => "Mutation Sig.",
            FeatureGroup::Clinical => "Age/Sex",
        }
    }
}

/// Convert a raw feature name to the canonical Open Nexus form.
///
/// Ports the final `standardize_feat_names` used by the raw-data inference
/// path (`get_onconpc_features_from_raw_data`):
/// - `XYZ_mut`  -> `XYZ`        (somatic mutation gene)
/// - `*AGE*`    -> `Age`
/// - `*GENDER*` -> `Sex`
/// - `SBS*`     -> unchanged
/// - otherwise  -> `"{name} CNA"`
pub fn standardize_feature_name(name: &str) -> String {
    if name.contains("_mut") {
        name.replace("_mut", "")
    } else if name.contains("AGE") || name.contains("Age") {
        "Age".to_string()
    } else if name.contains("GENDER") || name.contains("Sex") {
        "Sex".to_string()
    } else if name.contains("SBS") {
        name.to_string()
    } else {
        format!("{name} CNA")
    }
}

/// Partition feature names by group, preserving input order within groups.
pub fn partition_by_group(feature_names: &[String]) -> BTreeMap<FeatureGroup, Vec<String>> {
    let mut map: BTreeMap<FeatureGroup, Vec<String>> = BTreeMap::new();
    for name in feature_names {
        map.entry(FeatureGroup::classify(name))
            .or_default()
            .push(name.clone());
    }
    map
}

/// A dense, column-aligned matrix of feature values.
///
/// This is the cross-crate contract type (`nexus-genomics` produces it,
/// `nexus-ml` / `nexus-shap` consume it). It deliberately avoids a Polars
/// dependency so the core stays light; `nexus-data` converts to/from Polars
/// `DataFrame` at the I/O boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureMatrix {
    /// Row identifiers (sample IDs), length = `values.nrows()`.
    pub sample_ids: Vec<String>,
    /// Column names (feature names), length = `values.ncols()`.
    pub feature_names: Vec<String>,
    /// Row-major matrix of shape `[n_samples, n_features]`.
    pub values: Array2<f64>,
}

impl FeatureMatrix {
    /// Build a matrix, validating that the dimensions agree.
    pub fn new(
        sample_ids: Vec<String>,
        feature_names: Vec<String>,
        values: Array2<f64>,
    ) -> Result<Self> {
        if values.nrows() != sample_ids.len() {
            return Err(NexusError::ShapeMismatch(format!(
                "{} sample ids but {} rows",
                sample_ids.len(),
                values.nrows()
            )));
        }
        if values.ncols() != feature_names.len() {
            return Err(NexusError::ShapeMismatch(format!(
                "{} feature names but {} columns",
                feature_names.len(),
                values.ncols()
            )));
        }
        Ok(Self {
            sample_ids,
            feature_names,
            values,
        })
    }

    pub fn n_samples(&self) -> usize {
        self.sample_ids.len()
    }

    pub fn n_features(&self) -> usize {
        self.feature_names.len()
    }

    /// Index of a feature by name, if present.
    pub fn feature_index(&self, name: &str) -> Option<usize> {
        self.feature_names.iter().position(|f| f == name)
    }

    /// Reorder / pad columns so the matrix exactly matches `target`.
    ///
    /// Missing features are filled with zeros (ports `zero_pad_missing_features`
    /// followed by selection in the canonical order).
    pub fn align_to(&self, target: &[String]) -> Result<FeatureMatrix> {
        let mut values = Array2::<f64>::zeros((self.n_samples(), target.len()));
        for (new_col, fname) in target.iter().enumerate() {
            if let Some(old_col) = self.feature_index(fname) {
                for row in 0..self.n_samples() {
                    values[[row, new_col]] = self.values[[row, old_col]];
                }
            }
        }
        FeatureMatrix::new(self.sample_ids.clone(), target.to_vec(), values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_matches_python_order() {
        assert_eq!(FeatureGroup::classify("SBS1"), FeatureGroup::Signature);
        assert_eq!(FeatureGroup::classify("Age"), FeatureGroup::Clinical);
        assert_eq!(FeatureGroup::classify("Sex"), FeatureGroup::Clinical);
        assert_eq!(FeatureGroup::classify("ERBB2 CNA"), FeatureGroup::Cna);
        assert_eq!(FeatureGroup::classify("ERBB2"), FeatureGroup::Mutation);
    }

    #[test]
    fn standardize_names() {
        assert_eq!(standardize_feature_name("ERBB2_mut"), "ERBB2");
        assert_eq!(standardize_feature_name("AGE_AT_SEQ_REPORT"), "Age");
        assert_eq!(standardize_feature_name("GENDER"), "Sex");
        assert_eq!(standardize_feature_name("SBS1"), "SBS1");
        assert_eq!(standardize_feature_name("RAF1"), "RAF1 CNA");
    }

    #[test]
    fn align_pads_missing_with_zero() {
        let m = FeatureMatrix::new(
            vec!["s1".into()],
            vec!["A".into(), "B".into()],
            ndarray::array![[1.0, 2.0]],
        )
        .unwrap();
        let aligned = m.align_to(&["B".into(), "C".into(), "A".into()]).unwrap();
        assert_eq!(aligned.feature_names, vec!["B", "C", "A"]);
        assert_eq!(aligned.values, ndarray::array![[2.0, 0.0, 1.0]]);
    }
}
