//! CLI input parsing helpers: raw patient JSON and label tables.

use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::Deserialize;

use nexus_core::RawPatientInput;

#[derive(Deserialize)]
struct RawPatientJson {
    sample_id: String,
    age: f64,
    gender: String,
    #[serde(default)]
    cna_events: String,
    #[serde(default)]
    mutations: String,
}

/// Load an array of raw patient records from JSON.
pub fn load_raw_inputs(path: &PathBuf) -> Result<Vec<(String, RawPatientInput)>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading raw input {}", path.display()))?;
    let records: Vec<RawPatientJson> =
        serde_json::from_str(&text).context("raw input must be a JSON array of patient objects")?;
    Ok(records
        .into_iter()
        .map(|r| {
            (
                r.sample_id,
                RawPatientInput {
                    age: r.age,
                    gender: r.gender,
                    cna_events: r.cna_events,
                    mutations: r.mutations,
                },
            )
        })
        .collect())
}

/// Load integer class labels from a `cancer_label` column (.parquet or .tsv).
pub fn load_labels(path: &PathBuf) -> Result<Vec<usize>> {
    use polars::prelude::*;

    let is_parquet = path.extension().and_then(|e| e.to_str()) == Some("parquet");
    let df = if is_parquet {
        let file = std::fs::File::open(path)?;
        ParquetReader::new(file).finish()?
    } else {
        CsvReadOptions::default()
            .with_has_header(true)
            .with_parse_options(CsvParseOptions::default().with_separator(b'\t'))
            .try_into_reader_with_file_path(Some(path.clone()))?
            .finish()?
    };

    let col = df
        .column("cancer_label")
        .context("labels file must contain a 'cancer_label' column")?
        .cast(&DataType::Int64)?;
    let ca = col.i64()?;
    Ok(ca
        .into_iter()
        .map(|o| o.unwrap_or(0).max(0) as usize)
        .collect())
}
