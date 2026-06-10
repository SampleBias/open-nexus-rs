//! Open Nexus desktop engine and Tauri command handlers.

use std::sync::Arc;

use nexus_core::{CohortAgeStats, FeatureManifest, RawPatientInput};
use nexus_genomics::{FastaGenome, RawFeatureBuilder, SignatureSet};
use nexus_ml::Predictor;
use nexus_shap::Explainer;
use nexus_viz::{render_explanation_svg, ChartStyle};
use serde::Serialize;

/// Shared, loaded engine state.
pub struct Engine {
    predictor: Predictor,
    manifest: FeatureManifest,
    signatures: SignatureSet,
    genome: FastaGenome,
    age_stats: CohortAgeStats,
}

impl Engine {
    pub fn from_env() -> Result<Self, String> {
        let model = std::env::var("NEXUS_MODEL").map_err(|_| "NEXUS_MODEL not set")?;
        let metadata = std::env::var("NEXUS_METADATA").map_err(|_| "NEXUS_METADATA not set")?;
        let manifest = std::env::var("NEXUS_MANIFEST").map_err(|_| "NEXUS_MANIFEST not set")?;
        let predictor =
            Predictor::from_files(&model, &metadata).map_err(|e| e.to_string())?;
        let manifest = nexus_data::load_feature_manifest(&manifest).map_err(|e| e.to_string())?;
        let age_stats = match std::env::var("NEXUS_AGE_STATS") {
            Ok(p) => nexus_data::load_cohort_age_stats(&p).map_err(|e| e.to_string())?,
            Err(_) => CohortAgeStats {
                age_mean: 0.0,
                std_mean: 1.0,
            },
        };
        let signatures = match std::env::var("NEXUS_SIGNATURES") {
            Ok(d) => nexus_data::load_signature_dir(&d).map_err(|e| e.to_string())?,
            Err(_) => SignatureSet::default(),
        };
        let genome = match std::env::var("NEXUS_GENOME") {
            Ok(p) => FastaGenome::from_path(&p).map_err(|e| e.to_string())?,
            Err(_) => FastaGenome::default(),
        };
        Ok(Self {
            predictor,
            manifest,
            signatures,
            genome,
            age_stats,
        })
    }

    fn features(&self, input: RawPatientInput) -> Result<nexus_core::FeatureMatrix, String> {
        let builder = RawFeatureBuilder {
            genome: &self.genome,
            signatures: &self.signatures,
            manifest: &self.manifest,
            age_stats: self.age_stats,
        };
        builder
            .build(&[("sample".to_string(), input)])
            .map_err(|e| e.to_string())
    }
}

#[derive(Serialize)]
pub struct PredictResponse {
    pub predictions: Vec<nexus_core::SamplePrediction>,
}

#[derive(Serialize)]
pub struct ExplainResponse {
    pub explanation: nexus_core::ShapExplanation,
    pub svg: String,
}

pub fn predict(engine: Arc<Engine>, input: RawPatientInput) -> Result<PredictResponse, String> {
    let features = engine.features(input)?;
    let predictions = engine
        .predictor
        .predict_top_n(&features, 3)
        .map_err(|e| e.to_string())?;
    Ok(PredictResponse { predictions })
}

pub fn explain(engine: Arc<Engine>, input: RawPatientInput) -> Result<ExplainResponse, String> {
    let features = engine.features(input)?;
    let explainer = Explainer::new(engine.predictor.ensemble(), engine.predictor.metadata());
    let explanation = explainer
        .explain_sample(&features, 0, 10)
        .map_err(|e| e.to_string())?;
    let svg = render_explanation_svg(
        &explanation,
        Some(engine.age_stats),
        &ChartStyle::default(),
    );
    Ok(ExplainResponse { explanation, svg })
}
