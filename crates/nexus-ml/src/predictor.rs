//! High-level prediction API combining a [`TreeEnsemble`] with
//! [`ModelMetadata`] to produce ranked, named cancer-type predictions.

use nexus_core::{
    error::Result, FeatureMatrix, ModelMetadata, NexusError, Prediction, SamplePrediction,
};

use crate::tree::TreeEnsemble;

/// A loaded, ready-to-serve cancer-type classifier.
pub struct Predictor {
    ensemble: TreeEnsemble,
    metadata: ModelMetadata,
}

impl Predictor {
    /// Build a predictor, validating that the model's output dimension
    /// matches the number of declared target classes.
    pub fn new(ensemble: TreeEnsemble, metadata: ModelMetadata) -> Result<Self> {
        let n_out = ensemble.n_outputs();
        if n_out != metadata.target_classes.len() {
            return Err(NexusError::InvalidModel(format!(
                "model has {} outputs but metadata declares {} target classes",
                n_out,
                metadata.target_classes.len()
            )));
        }
        Ok(Self { ensemble, metadata })
    }

    /// Load model JSON + metadata JSON from disk.
    pub fn from_files(
        model_json: impl AsRef<std::path::Path>,
        metadata_json: impl AsRef<std::path::Path>,
    ) -> Result<Self> {
        let ensemble = TreeEnsemble::from_json_path(model_json)?;
        let meta_text = std::fs::read_to_string(metadata_json)?;
        let metadata: ModelMetadata = serde_json::from_str(&meta_text)?;
        Self::new(ensemble, metadata)
    }

    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    pub fn cancer_types(&self) -> &[String] {
        &self.metadata.target_classes
    }

    /// Predict ranked cancer types for every sample.
    ///
    /// The input matrix is aligned to the model's expected feature order
    /// first (zero-padding any missing columns), mirroring the Python
    /// pipeline which always feeds the canonical feature vector.
    pub fn predict(&self, features: &FeatureMatrix) -> Result<Vec<SamplePrediction>> {
        self.predict_top_n(features, self.cancer_types().len())
    }

    /// As [`Predictor::predict`] but keep only the top-`n` classes per sample.
    pub fn predict_top_n(
        &self,
        features: &FeatureMatrix,
        n: usize,
    ) -> Result<Vec<SamplePrediction>> {
        let aligned = features.align_to(&self.metadata.features)?;
        let classes = &self.metadata.target_classes;
        let mut out = Vec::with_capacity(aligned.n_samples());

        for (row_idx, sample_id) in aligned.sample_ids.iter().enumerate() {
            let row = aligned.values.row(row_idx);
            let probs = self.ensemble.predict_proba_row(row);

            // argmax for the headline prediction.
            let (best_idx, &max_posterior) = probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .ok_or_else(|| NexusError::invariant("empty probability vector"))?;

            let mut ranked: Vec<Prediction> = classes
                .iter()
                .zip(probs.iter())
                .map(|(c, &p)| Prediction {
                    cancer_type: c.clone(),
                    probability: p,
                })
                .collect();
            ranked.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap());
            ranked.truncate(n);

            out.push(SamplePrediction {
                sample_id: sample_id.clone(),
                cancer_type: classes[best_idx].clone(),
                max_posterior,
                predictions: ranked,
            });
        }
        Ok(out)
    }

    /// Borrow the underlying ensemble (used by `nexus-shap`).
    pub fn ensemble(&self) -> &TreeEnsemble {
        &self.ensemble
    }
}
