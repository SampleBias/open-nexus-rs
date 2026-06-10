//! # nexus-core
//!
//! Foundation crate for Open Nexus (Rust). Defines the shared domain types,
//! error type, feature taxonomy and configuration that every other crate
//! depends on. This crate is intentionally dependency-light (no Polars, no
//! async runtime) so it can be used everywhere — including the Tauri desktop
//! app and the Axum API — without pulling heavy transitive dependencies.
//!
//! Module map:
//! - [`error`]   — unified [`NexusError`] / [`Result`].
//! - [`feature`] — feature groups, name standardization, [`FeatureMatrix`].
//! - [`model`]   — [`ModelMetadata`], [`FeatureManifest`], [`CohortAgeStats`].
//! - [`domain`]  — predictions, SHAP explanations, input parsers.
//! - [`config`]  — [`NexusConfig`] (replaces `codes/config.yaml`).

pub mod config;
pub mod domain;
pub mod error;
pub mod feature;
pub mod model;

pub use config::{ColumnMapping, DataSource, NexusConfig};
pub use domain::{
    parse_cna_events, parse_mutations, CnaEvent, MutationRecord, Prediction, RawPatientInput,
    SamplePrediction, ShapExplanation, ShapFeature,
};
pub use error::{NexusError, Result};
pub use feature::{partition_by_group, standardize_feature_name, FeatureGroup, FeatureMatrix};
pub use model::{CohortAgeStats, FeatureManifest, ModelMetadata};
