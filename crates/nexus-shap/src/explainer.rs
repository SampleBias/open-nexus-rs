//! High-level explanation API: turn raw SHAP values into ranked, grouped
//! [`ShapExplanation`]s for the predicted class, mirroring
//! `get_onconpc_prediction_explanations` in the Python reference.

use nexus_core::{
    error::Result, FeatureGroup, FeatureMatrix, ModelMetadata, NexusError, ShapExplanation,
    ShapFeature,
};
use nexus_ml::tree::TreeEnsemble;

use crate::treeshap::ensemble_shap;

/// Explains predictions of a specific ensemble + metadata pair.
pub struct Explainer<'a> {
    ensemble: &'a TreeEnsemble,
    metadata: &'a ModelMetadata,
}

impl<'a> Explainer<'a> {
    pub fn new(ensemble: &'a TreeEnsemble, metadata: &'a ModelMetadata) -> Self {
        Self { ensemble, metadata }
    }

    /// Explain a single sample's *predicted* (argmax) class, returning the
    /// top-`top_n` features by absolute SHAP value.
    pub fn explain_sample(
        &self,
        features: &FeatureMatrix,
        sample_index: usize,
        top_n: usize,
    ) -> Result<ShapExplanation> {
        let aligned = features.align_to(&self.metadata.features)?;
        if sample_index >= aligned.n_samples() {
            return Err(NexusError::invariant(format!(
                "sample index {sample_index} out of range ({} samples)",
                aligned.n_samples()
            )));
        }
        let row = aligned.values.row(sample_index);

        let probs = self.ensemble.predict_proba_row(row);
        let (class_idx, &prob) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .ok_or_else(|| NexusError::invariant("empty probability vector"))?;

        let shap = ensemble_shap(self.ensemble, &row);
        let class_shap = &shap.values[class_idx];

        let mut feats: Vec<ShapFeature> = self
            .metadata
            .features
            .iter()
            .enumerate()
            .map(|(i, name)| ShapFeature {
                feature_name: name.clone(),
                group: FeatureGroup::classify(name),
                shap_value: class_shap.get(i).copied().unwrap_or(0.0),
                feature_value: aligned.values[[sample_index, i]],
            })
            .collect();

        feats.sort_by(|a, b| b.shap_value.abs().partial_cmp(&a.shap_value.abs()).unwrap());
        feats.truncate(top_n);

        Ok(ShapExplanation {
            sample_id: aligned.sample_ids[sample_index].clone(),
            predicted_cancer_type: self.metadata.target_classes[class_idx].clone(),
            predicted_probability: prob,
            features: feats,
        })
    }

    /// Explain every sample in the matrix.
    pub fn explain_all(
        &self,
        features: &FeatureMatrix,
        top_n: usize,
    ) -> Result<Vec<ShapExplanation>> {
        let aligned = features.align_to(&self.metadata.features)?;
        (0..aligned.n_samples())
            .map(|i| self.explain_sample(&aligned, i, top_n))
            .collect()
    }
}
