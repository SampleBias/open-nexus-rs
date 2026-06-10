//! Open Nexus desktop application (Tauri 2).
//!
//! Wraps the native engine (`nexus-ml` + `nexus-shap` + `nexus-genomics`) in a
//! small desktop shell. The frontend (in `ui/`) calls the `predict` and
//! `explain` commands; results and the SHAP SVG are rendered in-window.
//!
//! Model + artifact paths are provided via environment variables (same names as
//! the API server): `NEXUS_MODEL`, `NEXUS_METADATA`, `NEXUS_MANIFEST`,
//! `NEXUS_AGE_STATS`, `NEXUS_SIGNATURES`, `NEXUS_GENOME`.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;

use nexus_core::{CohortAgeStats, FeatureManifest, RawPatientInput};
use nexus_genomics::{FastaGenome, RawFeatureBuilder, SignatureSet};
use nexus_ml::Predictor;
use nexus_shap::Explainer;
use nexus_viz::{render_explanation_svg, ChartStyle};
use serde::Serialize;

/// Shared, loaded engine state.
struct Engine {
    predictor: Predictor,
    manifest: FeatureManifest,
    signatures: SignatureSet,
    genome: FastaGenome,
    age_stats: CohortAgeStats,
}

impl Engine {
    fn from_env() -> Result<Self, String> {
        let model = std::env::var("NEXUS_MODEL").map_err(|_| "NEXUS_MODEL not set")?;
        let metadata = std::env::var("NEXUS_METADATA").map_err(|_| "NEXUS_METADATA not set")?;
        let manifest = std::env::var("NEXUS_MANIFEST").map_err(|_| "NEXUS_MANIFEST not set")?;
        let predictor =
            Predictor::from_files(&model, &metadata).map_err(|e| e.to_string())?;
        let manifest = nexus_data::load_feature_manifest(&manifest).map_err(|e| e.to_string())?;
        let age_stats = match std::env::var("NEXUS_AGE_STATS") {
            Ok(p) => nexus_data::load_cohort_age_stats(&p).map_err(|e| e.to_string())?,
            Err(_) => CohortAgeStats { age_mean: 0.0, std_mean: 1.0 },
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
struct PredictResponse {
    predictions: Vec<nexus_core::SamplePrediction>,
}

#[derive(Serialize)]
struct ExplainResponse {
    explanation: nexus_core::ShapExplanation,
    svg: String,
}

#[tauri::command]
fn predict(
    engine: tauri::State<Arc<Engine>>,
    input: RawPatientInput,
) -> Result<PredictResponse, String> {
    let features = engine.features(input)?;
    let predictions = engine
        .predictor
        .predict_top_n(&features, 3)
        .map_err(|e| e.to_string())?;
    Ok(PredictResponse { predictions })
}

#[tauri::command]
fn explain(
    engine: tauri::State<Arc<Engine>>,
    input: RawPatientInput,
) -> Result<ExplainResponse, String> {
    let features = engine.features(input)?;
    let explainer = Explainer::new(engine.predictor.ensemble(), engine.predictor.metadata());
    let explanation = explainer
        .explain_sample(&features, 0, 10)
        .map_err(|e| e.to_string())?;
    let svg = render_explanation_svg(&explanation, Some(engine.age_stats), &ChartStyle::default());
    Ok(ExplainResponse { explanation, svg })
}

fn main() {
    let engine = Arc::new(Engine::from_env().expect("failed to load model artifacts"));
    tauri::Builder::default()
        .manage(engine)
        .invoke_handler(tauri::generate_handler![predict, explain])
        .run(tauri::generate_context!())
        .expect("error while running Open Nexus desktop");
}
