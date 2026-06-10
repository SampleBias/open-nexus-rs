//! Configuration types replacing `codes/config.yaml`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

use crate::error::Result;

/// Column-name mapping for a mutation source file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnMapping {
    pub sample_id: String,
    pub chr: String,
    pub pos: String,
    #[serde(rename = "ref")]
    pub reference: String,
    pub alt: String,
}

/// A single sequencing data source (GENIE or DFCI).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSource {
    pub file_path: String,
    pub columns: ColumnMapping,
    #[serde(default)]
    pub centers: Vec<String>,
}

/// Top-level configuration loaded from `config.yaml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NexusConfig {
    #[serde(default)]
    pub base_path: Option<String>,
    #[serde(default)]
    pub data_sources: BTreeMap<String, DataSource>,
}

impl NexusConfig {
    /// Load configuration from a YAML file.
    pub fn from_yaml_path(path: impl AsRef<Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_yaml::from_str(&text)?)
    }

    /// Parse configuration from a YAML string.
    pub fn from_yaml_str(text: &str) -> Result<Self> {
        Ok(serde_yaml::from_str(text)?)
    }
}
