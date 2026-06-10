//! # nexus-ml
//!
//! Native machine-learning runtime for Open Nexus.
//!
//! - [`tree`]: parse XGBoost `save_model` JSON into a [`TreeEnsemble`] and run
//!   inference natively (no XGBoost C dependency). This same tree structure is
//!   consumed by `nexus-shap` for exact Tree SHAP.
//! - [`predictor`]: [`Predictor`] couples an ensemble with [`nexus_core::ModelMetadata`]
//!   to emit ranked, named [`nexus_core::SamplePrediction`]s.
//! - [`filter`]: low-frequency feature/sample filtering.
//! - [`cv`]: stratified k-fold splitting + per-fold age normalization.
//! - [`metrics`]: sklearn-style classification report + posterior cutoffs.
//!
//! Training the booster itself (gradient descent over trees) is intentionally
//! out of the default build; the surrounding CV/filtering/metrics orchestration
//! lives here so a training driver can bind any booster backend.

pub mod cv;
pub mod filter;
pub mod metrics;
pub mod predictor;
pub mod tree;

pub use cv::{apply_age_norm, fit_age_norm, stratified_kfold, AgeNorm};
pub use filter::{low_frequency_filter, FilterOutcome, FilterThresholds};
pub use metrics::{
    classification_report, filter_by_posterior_cutoff, ClassMetrics, ClassificationReport,
};
pub use predictor::Predictor;
pub use tree::{softmax, Tree, TreeEnsemble};
