//! Integration tests for the desktop app's Tauri command logic.

use std::sync::Arc;

use nexus_core::RawPatientInput;
use nexus_desktop::Engine;

fn example_engine() -> Arc<Engine> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    std::env::set_var("NEXUS_MODEL", root.join("examples/model.json"));
    std::env::set_var("NEXUS_METADATA", root.join("examples/metadata.json"));
    std::env::set_var("NEXUS_MANIFEST", root.join("examples/manifest.json"));
    std::env::set_var("NEXUS_AGE_STATS", root.join("examples/age_stats.json"));
    std::env::set_var("NEXUS_GENOME", root.join("examples/genome.fa"));
    Arc::new(Engine::from_env().expect("engine should load example artifacts"))
}

fn sample_input() -> RawPatientInput {
    RawPatientInput {
        age: 72.0,
        gender: "male".into(),
        cna_events: "RAF1 2 | PPARG 2".into(),
        mutations: "ERBB2, chr17, 10, C, T".into(),
    }
}

#[test]
fn predict_returns_top_cancer_types() {
    let engine = example_engine();
    let resp = nexus_desktop::predict(engine, sample_input()).expect("predict should succeed");
    assert_eq!(resp.predictions.len(), 1);
    let sample = &resp.predictions[0];
    assert_eq!(sample.cancer_type, "Non-Small Cell Lung Cancer");
    assert!((sample.max_posterior - 0.9241418199787566).abs() < 1e-6);
    assert_eq!(sample.predictions.len(), 2);
}

#[test]
fn explain_returns_shap_and_svg() {
    let engine = example_engine();
    let resp = nexus_desktop::explain(engine, sample_input()).expect("explain should succeed");
    assert_eq!(
        resp.explanation.predicted_cancer_type,
        "Non-Small Cell Lung Cancer"
    );
    assert!(!resp.explanation.features.is_empty());
    assert!(resp.svg.contains("<svg"));
    assert!(resp.svg.contains("ERBB2"));
}
