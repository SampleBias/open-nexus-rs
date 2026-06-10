//! Model-side artifacts: metadata, canonical feature manifest, cohort stats.
//!
//! These replace the Python pickle/JSON artifacts (`features_onconpc.pkl`,
//! `combined_cohort_age_stats.pkl`, `model_metadata.json`) with explicit,
//! serde-friendly Rust types. The `pickle-migrate` tool emits JSON matching
//! these schemas.

use serde::{Deserialize, Serialize};

use crate::error::{NexusError, Result};

/// Companion metadata shipped alongside a trained XGBoost model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    /// Ordered feature names the model expects as input columns.
    pub features: Vec<String>,
    /// Ordered class labels (cancer types) the model can output.
    pub target_classes: Vec<String>,
}

impl ModelMetadata {
    /// Validate that a feature matrix's columns are a superset of the
    /// required features (ports the spirit of `validate_model`).
    pub fn validate_features(&self, present: &[String]) -> Result<()> {
        for required in &self.features {
            if !present.iter().any(|p| p == required) {
                return Err(NexusError::feature_mismatch(format!(
                    "missing required feature '{required}'"
                )));
            }
        }
        Ok(())
    }
}

/// Canonical feature ordering used to zero-pad/align every sample
/// (the contents of `features_onconpc.pkl`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureManifest {
    pub features: Vec<String>,
}

impl FeatureManifest {
    pub fn new(features: Vec<String>) -> Self {
        Self { features }
    }

    pub fn len(&self) -> usize {
        self.features.len()
    }

    pub fn is_empty(&self) -> bool {
        self.features.is_empty()
    }
}

/// Cohort age normalization statistics (`combined_cohort_age_stats.pkl`).
///
/// Field names match the Python pickle keys for a faithful migration.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CohortAgeStats {
    #[serde(rename = "Age_mean")]
    pub age_mean: f64,
    #[serde(rename = "Std_mean")]
    pub std_mean: f64,
}

impl CohortAgeStats {
    /// Standardize a raw age value: `(age - mean) / std`.
    pub fn standardize(&self, age: f64) -> f64 {
        (age - self.age_mean) / self.std_mean
    }

    /// Invert standardization for display: `age * std + mean`.
    pub fn denormalize(&self, z: f64) -> f64 {
        z * self.std_mean + self.age_mean
    }
}
