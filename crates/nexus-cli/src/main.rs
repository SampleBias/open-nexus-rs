//! `nexus` — the Open Nexus command-line interface.
//!
//! Subcommands mirror the original Python workflow:
//! - `process-features`: GENIE files -> aligned feature matrix (Parquet).
//! - `predict`: feature matrix or raw patient JSON -> ranked cancer types.
//! - `explain`: predictions + native Tree SHAP -> JSON + SVG charts.
//! - `train`: CV/filtering orchestration report over features + labels.

mod input;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};

use nexus_core::FeatureMatrix;
use nexus_data as data;
use nexus_genomics::{FastaGenome, RawFeatureBuilder};
use nexus_ml::{low_frequency_filter, stratified_kfold, FilterThresholds, Predictor};
use nexus_shap::Explainer;
use nexus_viz::{render_explanation_svg, ChartStyle};

#[derive(Parser)]
#[command(
    name = "nexus",
    version,
    about = "Open Nexus — cancer-of-unknown-primary classifier"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Assemble a model-ready feature matrix from GENIE files.
    ProcessFeatures(ProcessFeaturesArgs),
    /// Predict cancer types from a feature matrix or raw patient input.
    Predict(PredictArgs),
    /// Explain predictions with native Tree SHAP (JSON + SVG).
    Explain(ExplainArgs),
    /// Run CV/filtering orchestration over features + labels.
    Train(TrainArgs),
}

#[derive(Parser)]
struct ProcessFeaturesArgs {
    #[arg(long)]
    mutations: PathBuf,
    #[arg(long)]
    patients: PathBuf,
    #[arg(long)]
    samples: PathBuf,
    #[arg(long)]
    signatures: PathBuf,
    #[arg(long)]
    genome: PathBuf,
    #[arg(long)]
    manifest: PathBuf,
    #[arg(long)]
    age_stats: PathBuf,
    #[arg(long)]
    out: PathBuf,
}

#[derive(Parser)]
struct PredictArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    metadata: PathBuf,
    /// Pre-built feature matrix (.parquet or .tsv).
    #[arg(long, conflicts_with = "raw")]
    features: Option<PathBuf>,
    /// Raw patient JSON (array of {sample_id, age, gender, cna_events, mutations}).
    #[arg(long, requires = "manifest")]
    raw: Option<PathBuf>,
    #[arg(long)]
    manifest: Option<PathBuf>,
    #[arg(long)]
    age_stats: Option<PathBuf>,
    #[arg(long)]
    signatures: Option<PathBuf>,
    #[arg(long)]
    genome: Option<PathBuf>,
    /// Keep top-N classes per sample (default: all).
    #[arg(long)]
    top_n: Option<usize>,
}

#[derive(Parser)]
struct ExplainArgs {
    #[arg(long)]
    model: PathBuf,
    #[arg(long)]
    metadata: PathBuf,
    #[arg(long, conflicts_with = "raw")]
    features: Option<PathBuf>,
    #[arg(long, requires = "manifest")]
    raw: Option<PathBuf>,
    #[arg(long)]
    manifest: Option<PathBuf>,
    #[arg(long)]
    age_stats: Option<PathBuf>,
    #[arg(long)]
    signatures: Option<PathBuf>,
    #[arg(long)]
    genome: Option<PathBuf>,
    #[arg(long, default_value_t = 10)]
    top_n: usize,
    /// Directory to write one SVG chart per sample.
    #[arg(long)]
    out_dir: Option<PathBuf>,
}

#[derive(Parser)]
struct TrainArgs {
    #[arg(long)]
    features: PathBuf,
    #[arg(long)]
    labels: PathBuf,
    #[arg(long, default_value_t = 10)]
    k_fold: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::ProcessFeatures(a) => process_features(a),
        Command::Predict(a) => predict(a),
        Command::Explain(a) => explain(a),
        Command::Train(a) => train(a),
    }
}

/// Load a feature matrix from a `.parquet` or `.tsv` file.
fn load_features(path: &PathBuf) -> Result<FeatureMatrix> {
    let is_parquet = path.extension().and_then(|e| e.to_str()) == Some("parquet");
    if is_parquet {
        Ok(data::read_parquet(path)?)
    } else {
        let df = data::read_tsv(path)?;
        Ok(data::frame_to_matrix(&df)?)
    }
}

/// Build a feature matrix from raw patient JSON.
fn build_raw_features(
    raw: &PathBuf,
    manifest: &PathBuf,
    age_stats: &Option<PathBuf>,
    signatures: &Option<PathBuf>,
    genome: &Option<PathBuf>,
) -> Result<FeatureMatrix> {
    let inputs = input::load_raw_inputs(raw)?;
    let manifest = data::load_feature_manifest(manifest)?;
    let age = match age_stats {
        Some(p) => data::load_cohort_age_stats(p)?,
        None => nexus_core::CohortAgeStats {
            age_mean: 0.0,
            std_mean: 1.0,
        },
    };
    let sigs = match signatures {
        Some(dir) => data::load_signature_dir(dir)?,
        None => nexus_genomics::SignatureSet::default(),
    };
    let genome = match genome {
        Some(p) => FastaGenome::from_path(p)?,
        None => FastaGenome::default(),
    };
    let builder = RawFeatureBuilder {
        genome: &genome,
        signatures: &sigs,
        manifest: &manifest,
        age_stats: age,
    };
    Ok(builder.build(&inputs)?)
}

fn resolve_features(
    features: &Option<PathBuf>,
    raw: &Option<PathBuf>,
    manifest: &Option<PathBuf>,
    age_stats: &Option<PathBuf>,
    signatures: &Option<PathBuf>,
    genome: &Option<PathBuf>,
) -> Result<FeatureMatrix> {
    match (features, raw) {
        (Some(f), _) => load_features(f),
        (None, Some(r)) => {
            let manifest = manifest.as_ref().context("--raw requires --manifest")?;
            build_raw_features(r, &manifest.clone(), age_stats, signatures, genome)
        }
        (None, None) => anyhow::bail!("provide either --features or --raw"),
    }
}

fn process_features(a: ProcessFeaturesArgs) -> Result<()> {
    let rows = data::load_genie_mutations(&a.mutations)?;
    let events = data::to_mutation_events(&rows);
    let snvs = data::to_snv_calls(&rows);
    let clinical = data::load_genie_clinical(&a.patients, &a.samples)?;

    let genome = FastaGenome::from_path(&a.genome)?;
    let sigs = data::load_signature_dir(&a.signatures)?;
    let manifest = data::load_feature_manifest(&a.manifest)?;
    let age = data::load_cohort_age_stats(&a.age_stats)?;

    let (sig_samples, channels, counts) = nexus_genomics::build_sbs96(&snvs, &genome)?;
    let (sig_names, sig_matrix) = sigs.project(&channels, &counts)?;

    let cna: Vec<nexus_genomics::CnaValue> = Vec::new(); // GENIE CNA wiring is a follow-up
    let assembled = nexus_genomics::assemble_feature_matrix(
        &events,
        &cna,
        &sig_samples,
        &sig_names,
        &sig_matrix,
        &clinical,
    )?;

    let mut aligned = assembled.align_to(&manifest.features)?;
    if let Some(age_col) = aligned.feature_index("Age") {
        for r in 0..aligned.n_samples() {
            let v = aligned.values[[r, age_col]];
            aligned.values[[r, age_col]] = age.standardize(v);
        }
    }
    data::write_parquet(&aligned, &a.out)?;
    tracing::info!(
        "wrote {} samples x {} features to {}",
        aligned.n_samples(),
        aligned.n_features(),
        a.out.display()
    );
    Ok(())
}

fn predict(a: PredictArgs) -> Result<()> {
    let features = resolve_features(
        &a.features,
        &a.raw,
        &a.manifest,
        &a.age_stats,
        &a.signatures,
        &a.genome,
    )?;
    let predictor = Predictor::from_files(&a.model, &a.metadata)?;
    let n = a.top_n.unwrap_or_else(|| predictor.cancer_types().len());
    let preds = predictor.predict_top_n(&features, n)?;
    println!("{}", serde_json::to_string_pretty(&preds)?);
    Ok(())
}

fn explain(a: ExplainArgs) -> Result<()> {
    let features = resolve_features(
        &a.features,
        &a.raw,
        &a.manifest,
        &a.age_stats,
        &a.signatures,
        &a.genome,
    )?;
    let predictor = Predictor::from_files(&a.model, &a.metadata)?;
    let explainer = Explainer::new(predictor.ensemble(), predictor.metadata());
    let explanations = explainer.explain_all(&features, a.top_n)?;

    let age = match &a.age_stats {
        Some(p) => Some(data::load_cohort_age_stats(p)?),
        None => None,
    };

    if let Some(dir) = &a.out_dir {
        std::fs::create_dir_all(dir)?;
        for ex in &explanations {
            let svg = render_explanation_svg(ex, age, &ChartStyle::default());
            let path = dir.join(format!("{}.svg", sanitize(&ex.sample_id)));
            std::fs::write(&path, svg)?;
            tracing::info!("wrote {}", path.display());
        }
    }
    println!("{}", serde_json::to_string_pretty(&explanations)?);
    Ok(())
}

fn train(a: TrainArgs) -> Result<()> {
    let features = load_features(&a.features)?;
    let labels = input::load_labels(&a.labels)?;
    if labels.len() != features.n_samples() {
        anyhow::bail!(
            "labels ({}) and feature rows ({}) differ",
            labels.len(),
            features.n_samples()
        );
    }
    let filtered = low_frequency_filter(&features, FilterThresholds::default())?;
    tracing::info!(
        "after low-frequency filter: {} samples, {} features ({} dropped features, {} dropped samples)",
        filtered.matrix.n_samples(),
        filtered.matrix.n_features(),
        filtered.dropped_features.len(),
        filtered.dropped_samples.len()
    );

    // Re-derive labels for retained samples.
    let kept_labels: Vec<usize> = (0..features.n_samples())
        .filter(|i| !filtered.dropped_samples.contains(i))
        .map(|i| labels[i])
        .collect();
    let folds = stratified_kfold(&kept_labels, a.k_fold, a.seed)?;

    let mut fold_sizes = vec![0usize; a.k_fold];
    for &f in &folds {
        fold_sizes[f] += 1;
    }
    let report = serde_json::json!({
        "n_samples_after_filter": filtered.matrix.n_samples(),
        "n_features_after_filter": filtered.matrix.n_features(),
        "dropped_features": filtered.dropped_features,
        "k_fold": a.k_fold,
        "fold_sizes": fold_sizes,
        "note": "Booster fitting requires a gradient-boosting backend; CV split and \
                 filtering are produced here. Inference/SHAP run natively on saved XGBoost JSON."
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}
