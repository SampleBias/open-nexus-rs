//! Shared application state and the feature-building abstraction.

use std::sync::Arc;

use nexus_core::{CohortAgeStats, FeatureManifest, FeatureMatrix, RawPatientInput};
use nexus_genomics::{FastaGenome, RawFeatureBuilder, SignatureSet};
use nexus_ml::Predictor;

use crate::error::ApiError;
use crate::storage::Storage;

/// Turns raw patient inputs into a model-ready feature matrix.
pub trait FeatureSource: Send + Sync {
    fn build(&self, inputs: &[(String, RawPatientInput)]) -> Result<FeatureMatrix, ApiError>;
}

/// Production feature source backed by genome + signatures + manifest.
pub struct RawFeatureSource {
    pub genome: FastaGenome,
    pub signatures: SignatureSet,
    pub manifest: FeatureManifest,
    pub age_stats: CohortAgeStats,
}

impl FeatureSource for RawFeatureSource {
    fn build(&self, inputs: &[(String, RawPatientInput)]) -> Result<FeatureMatrix, ApiError> {
        let builder = RawFeatureBuilder {
            genome: &self.genome,
            signatures: &self.signatures,
            manifest: &self.manifest,
            age_stats: self.age_stats,
        };
        Ok(builder.build(inputs)?)
    }
}

/// Cloneable application state shared across handlers.
#[derive(Clone)]
pub struct AppState {
    pub predictor: Option<Arc<Predictor>>,
    pub feature_source: Option<Arc<dyn FeatureSource>>,
    pub storage: Arc<dyn Storage>,
}

impl AppState {
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self {
            predictor: None,
            feature_source: None,
            storage,
        }
    }

    pub fn with_model(
        mut self,
        predictor: Arc<Predictor>,
        feature_source: Arc<dyn FeatureSource>,
    ) -> Self {
        self.predictor = Some(predictor);
        self.feature_source = Some(feature_source);
        self
    }

    pub fn model_loaded(&self) -> bool {
        self.predictor.is_some() && self.feature_source.is_some()
    }
}
