//! Load mutation-signature weight CSVs into a [`SignatureSet`].
//!
//! Ports the file parsing inside `obtain_mutation_signatures`: each CSV has a
//! `Type` column (substitution, e.g. `C>A`), a `Subtype` column (trinucleotide,
//! e.g. `ACA`), and one or more weight columns. The channel label is built as
//! `"{Subtype[0]}[{Type}]{Subtype[2]}"`; the weight column preferring a
//! GRCh38/GRCh37 human reference is used.

use polars::prelude::*;

use nexus_core::{error::Result, NexusError};
use nexus_genomics::{Signature, SignatureSet};

fn polars_err(e: PolarsError) -> NexusError {
    NexusError::Invariant(format!("polars: {e}"))
}

/// Load a single signature CSV file. The signature name defaults to the file
/// stem unless `name` is provided.
pub fn load_signature_csv(
    path: impl AsRef<std::path::Path>,
    name: Option<String>,
) -> Result<Signature> {
    let path = path.as_ref();
    let sig_name = name.unwrap_or_else(|| {
        path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("signature")
            .to_string()
    });

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(path.to_path_buf()))
        .map_err(polars_err)?
        .finish()
        .map_err(polars_err)?;

    let type_col = df
        .column("Type")
        .map_err(polars_err)?
        .str()
        .map_err(polars_err)?;
    let subtype_col = df
        .column("Subtype")
        .map_err(polars_err)?
        .str()
        .map_err(polars_err)?;

    // Choose the weight column: prefer GRCh38, then GRCh37, else first numeric.
    let weight_name = pick_weight_column(&df)?;
    let weight_col = df
        .column(&weight_name)
        .map_err(polars_err)?
        .cast(&DataType::Float64)
        .map_err(polars_err)?;
    let weight_col = weight_col.f64().map_err(polars_err)?;

    let mut weights = std::collections::BTreeMap::new();
    for ((t, st), w) in type_col
        .into_iter()
        .zip(subtype_col.into_iter())
        .zip(weight_col.into_iter())
    {
        let (t, st) = match (t, st) {
            (Some(t), Some(st)) if st.len() >= 3 => (t, st),
            _ => continue,
        };
        let w = w.unwrap_or(0.0);
        let sb = st.as_bytes();
        let channel = format!("{}[{}]{}", sb[0] as char, t, sb[2] as char);
        weights.insert(channel, w);
    }

    Ok(Signature {
        name: sig_name,
        weights,
    })
}

fn pick_weight_column(df: &DataFrame) -> Result<String> {
    let names: Vec<String> = df
        .get_column_names()
        .iter()
        .map(|s| s.to_string())
        .collect();
    if let Some(n) = names.iter().find(|n| n.contains("GRCh38")) {
        return Ok(n.clone());
    }
    if let Some(n) = names.iter().find(|n| n.contains("GRCh37")) {
        return Ok(n.clone());
    }
    for n in &names {
        if n == "Type" || n == "Subtype" {
            continue;
        }
        let dt = df.column(n).map_err(polars_err)?.dtype();
        if dt.is_primitive_numeric() {
            return Ok(n.clone());
        }
    }
    Err(NexusError::invariant(
        "no numeric weight column found in signature CSV",
    ))
}

/// Load every `*.csv` in a directory into a [`SignatureSet`].
pub fn load_signature_dir(dir: impl AsRef<std::path::Path>) -> Result<SignatureSet> {
    let mut signatures = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) == Some("csv") {
            signatures.push(load_signature_csv(&path, None)?);
        }
    }
    signatures.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(SignatureSet::new(signatures))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn loads_a_minimal_signature() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("SBS1.csv");
        let mut f = std::fs::File::create(&path).unwrap();
        // Two channels with GRCh38 weights.
        writeln!(f, "Type,Subtype,SBS1_GRCh38").unwrap();
        writeln!(f, "C>A,ACA,0.4").unwrap();
        writeln!(f, "C>T,TCG,0.6").unwrap();
        drop(f);

        let sig = load_signature_csv(&path, None).unwrap();
        assert_eq!(sig.name, "SBS1");
        assert_eq!(sig.weights.get("A[C>A]A"), Some(&0.4));
        assert_eq!(sig.weights.get("T[C>T]G"), Some(&0.6));
    }
}
