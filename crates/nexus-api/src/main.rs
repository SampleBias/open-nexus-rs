//! `nexus-api` server binary.
//!
//! Configuration via environment variables (all optional except bind address):
//! - `NEXUS_BIND`        (default `0.0.0.0:5000`)
//! - `NEXUS_MODEL`       XGBoost model JSON
//! - `NEXUS_METADATA`    model metadata JSON
//! - `NEXUS_MANIFEST`    canonical feature manifest JSON
//! - `NEXUS_AGE_STATS`   cohort age stats JSON
//! - `NEXUS_SIGNATURES`  directory of signature weight CSVs
//! - `NEXUS_GENOME`      reference FASTA
//!
//! When the model variables are present the prediction endpoint is enabled;
//! otherwise auth/history/health still work and predict returns 503.

use std::sync::Arc;

use anyhow::{Context, Result};

use nexus_api::state::RawFeatureSource;
use nexus_api::{build_router, AppState, InMemoryStorage};
use nexus_data as data;
use nexus_genomics::FastaGenome;
use nexus_ml::Predictor;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let storage = Arc::new(InMemoryStorage::new());
    let mut state = AppState::new(storage);

    if let (Ok(model), Ok(metadata), Ok(manifest)) = (
        std::env::var("NEXUS_MODEL"),
        std::env::var("NEXUS_METADATA"),
        std::env::var("NEXUS_MANIFEST"),
    ) {
        tracing::info!("loading model artifacts");
        let predictor =
            Predictor::from_files(&model, &metadata).context("loading model + metadata")?;
        let manifest = data::load_feature_manifest(&manifest)?;
        let age_stats = match std::env::var("NEXUS_AGE_STATS") {
            Ok(p) => data::load_cohort_age_stats(&p)?,
            Err(_) => nexus_core::CohortAgeStats {
                age_mean: 0.0,
                std_mean: 1.0,
            },
        };
        let signatures = match std::env::var("NEXUS_SIGNATURES") {
            Ok(dir) => data::load_signature_dir(&dir)?,
            Err(_) => nexus_genomics::SignatureSet::default(),
        };
        let genome = match std::env::var("NEXUS_GENOME") {
            Ok(p) => FastaGenome::from_path(&p)?,
            Err(_) => FastaGenome::default(),
        };
        let feature_source = Arc::new(RawFeatureSource {
            genome,
            signatures,
            manifest,
            age_stats,
        });
        state = state.with_model(Arc::new(predictor), feature_source);
    } else {
        tracing::warn!("NEXUS_MODEL/METADATA/MANIFEST not all set; predict endpoint disabled");
    }

    let bind = std::env::var("NEXUS_BIND").unwrap_or_else(|_| "0.0.0.0:5000".into());
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .with_context(|| format!("binding {bind}"))?;
    tracing::info!("Open Nexus API listening on {bind}");
    axum::serve(listener, build_router(state)).await?;
    Ok(())
}
