//! # nexus-shap
//!
//! Exact, native Tree SHAP explanations for Open Nexus models. No Python, no
//! `shap` dependency — the algorithm operates directly on the
//! [`nexus_ml::tree::TreeEnsemble`] structure.
//!
//! - [`treeshap`]: the core polynomial-time algorithm + base-value computation.
//! - [`explainer`]: [`Explainer`] produces ranked, grouped
//!   [`nexus_core::ShapExplanation`]s for the predicted class.

pub mod explainer;
pub mod treeshap;

pub use explainer::Explainer;
pub use treeshap::{ensemble_shap, tree_expected_value, tree_shap, EnsembleShap};
