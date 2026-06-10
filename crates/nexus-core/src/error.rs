//! Unified error type shared across all Open Nexus crates.

use thiserror::Error;

/// Result alias used throughout the workspace.
pub type Result<T> = std::result::Result<T, NexusError>;

/// All error conditions surfaced by the Open Nexus core engine.
#[derive(Debug, Error)]
pub enum NexusError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON (de)serialization error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("YAML (de)serialization error: {0}")]
    Yaml(#[from] serde_yaml::Error),

    /// A feature matrix did not contain the columns the model requires.
    #[error("feature mismatch: {0}")]
    FeatureMismatch(String),

    /// Dimensions of two structures that must agree did not.
    #[error("shape mismatch: {0}")]
    ShapeMismatch(String),

    /// A user-supplied string (CNA / mutation spec) could not be parsed.
    #[error("parse error: {0}")]
    Parse(String),

    /// A required artifact (model, metadata, signature weights) was missing.
    #[error("missing artifact: {0}")]
    MissingArtifact(String),

    /// A model file was structurally invalid.
    #[error("invalid model: {0}")]
    InvalidModel(String),

    /// Catch-all for validated invariants.
    #[error("{0}")]
    Invariant(String),
}

impl NexusError {
    pub fn parse(msg: impl Into<String>) -> Self {
        NexusError::Parse(msg.into())
    }

    pub fn invariant(msg: impl Into<String>) -> Self {
        NexusError::Invariant(msg.into())
    }

    pub fn feature_mismatch(msg: impl Into<String>) -> Self {
        NexusError::FeatureMismatch(msg.into())
    }
}
