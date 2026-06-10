//! End-to-end pipeline test: raw patient input -> genomic feature assembly ->
//! native XGBoost inference -> Tree SHAP -> SVG explanation. Exercises every
//! core crate together without any Python or external data.

use ndarray::array;

use nexus_core::{CohortAgeStats, FeatureManifest, ModelMetadata, RawPatientInput};
use nexus_genomics::{FastaGenome, RawFeatureBuilder, Signature, SignatureSet};
use nexus_ml::{Predictor, TreeEnsemble};
use nexus_shap::{ensemble_shap, Explainer};
use nexus_viz::{render_explanation_svg, ChartStyle};

/// A 2-class model over [ERBB2, RAF1 CNA, SBS1, Age, Sex]:
/// - class 0 ("NSCLC"): stump on ERBB2 (index 0) -> present pushes toward NSCLC
/// - class 1 ("CRC"):   stump on Age   (index 3) -> high age pushes toward CRC
fn model_json() -> String {
    serde_json::json!({
        "learner": {
            "feature_names": ["ERBB2", "RAF1 CNA", "SBS1", "Age", "Sex"],
            "learner_model_param": {"base_score":"5E-1","num_class":"2","num_feature":"5"},
            "gradient_booster": {"model": {
                "tree_info": [0, 1],
                "trees": [
                    {"left_children":[1,-1,-1],"right_children":[2,-1,-1],
                     "split_indices":[0,0,0],"split_conditions":[0.5,-1.0,1.5],
                     "default_left":[1,0,0],"sum_hessian":[20.0,10.0,10.0]},
                    {"left_children":[1,-1,-1],"right_children":[2,-1,-1],
                     "split_indices":[3,0,0],"split_conditions":[1.0,-1.0,1.5],
                     "default_left":[1,0,0],"sum_hessian":[20.0,10.0,10.0]}
                ]
            }}
        }
    })
    .to_string()
}

#[test]
fn full_pipeline_raw_to_explanation() {
    // Reference genome: chr17 with a known base context around position 10.
    // positions: 1:A 2:C 3:G 4:T 5:A 6:C 7:G 8:T 9:A 10:C 11:A 12:G ...
    let genome = FastaGenome::parse(">chr17\nACGTACGTACAG\n");

    // One uniform signature so projection is well-defined.
    let channels = nexus_genomics::sbs96_channels();
    let w = 1.0 / channels.len() as f64;
    let sig = Signature {
        name: "SBS1".into(),
        weights: channels.into_iter().map(|c| (c, w)).collect(),
    };
    let signatures = SignatureSet::new(vec![sig]);

    let manifest = FeatureManifest::new(vec![
        "ERBB2".into(),
        "RAF1 CNA".into(),
        "SBS1".into(),
        "Age".into(),
        "Sex".into(),
    ]);
    let age_stats = CohortAgeStats {
        age_mean: 0.0,
        std_mean: 1.0,
    };

    let builder = RawFeatureBuilder {
        genome: &genome,
        signatures: &signatures,
        manifest: &manifest,
        age_stats,
    };

    // A patient with an ERBB2 SNV at chr17:10 (C>T), a RAF1 amplification, age 70.
    let inputs = vec![(
        "PATIENT-1".to_string(),
        RawPatientInput {
            age: 70.0,
            gender: "male".into(),
            cna_events: "RAF1 2".into(),
            mutations: "ERBB2, chr17, 10, C, T".into(),
        },
    )];

    let features = builder.build(&inputs).expect("feature build");
    assert_eq!(features.feature_names, manifest.features);
    assert_eq!(features.n_samples(), 1);
    // ERBB2 mutation counted.
    let erbb2 = features.feature_index("ERBB2").unwrap();
    assert_eq!(features.values[[0, erbb2]], 1.0);
    // RAF1 CNA value carried through.
    let raf1 = features.feature_index("RAF1 CNA").unwrap();
    assert_eq!(features.values[[0, raf1]], 2.0);

    // Inference.
    let ensemble = TreeEnsemble::from_json_slice(model_json().as_bytes()).unwrap();
    let metadata = ModelMetadata {
        features: manifest.features.clone(),
        target_classes: vec!["NSCLC".into(), "CRC".into()],
    };
    let predictor = Predictor::new(ensemble, metadata.clone()).unwrap();
    let preds = predictor.predict(&features).unwrap();
    assert_eq!(preds.len(), 1);
    let probs_sum: f64 = preds[0].predictions.iter().map(|p| p.probability).sum();
    assert!((probs_sum - 1.0).abs() < 1e-9);

    // SHAP local accuracy on the assembled sample.
    let row = features.values.row(0);
    let shap = ensemble_shap(predictor.ensemble(), &row);
    let margins = predictor.ensemble().predict_margins(row);
    for (c, &margin) in margins.iter().enumerate() {
        let recon: f64 = shap.values[c].iter().sum::<f64>() + shap.base_values[c];
        assert!((recon - margin).abs() < 1e-9);
    }

    // Explanation + SVG.
    let explainer = Explainer::new(predictor.ensemble(), predictor.metadata());
    let explanation = explainer.explain_sample(&features, 0, 5).unwrap();
    assert_eq!(explanation.sample_id, "PATIENT-1");
    let svg = render_explanation_svg(&explanation, Some(age_stats), &ChartStyle::default());
    assert!(svg.contains("PATIENT-1"));
    assert!(svg.starts_with("<svg"));
}

#[test]
fn sanity_check_array_macro_available() {
    // Guard: ensures ndarray dev-dep wiring stays intact.
    let a = array![1.0, 2.0, 3.0];
    assert_eq!(a.sum(), 6.0);
}
