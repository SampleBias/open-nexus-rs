//! AACR GENIE file loaders (mutations + clinical), porting the column
//! selection in `process_features.py::load_genie_data`.

use polars::prelude::*;

use nexus_core::{error::Result, NexusError};
use nexus_genomics::{PatientClinical, SnvCall};

fn polars_err(e: PolarsError) -> NexusError {
    NexusError::Invariant(format!("polars: {e}"))
}

/// One mutation row needed for both gene-count and SBS96 features.
#[derive(Debug, Clone)]
pub struct MutationRow {
    pub sample_id: String,
    pub gene: String,
    pub chromosome: String,
    pub position: u64,
    pub reference_allele: String,
    pub alternate_allele: String,
}

fn read_table(path: impl AsRef<std::path::Path>) -> Result<DataFrame> {
    CsvReadOptions::default()
        .with_has_header(true)
        .with_parse_options(
            CsvParseOptions::default()
                .with_separator(b'\t')
                .with_comment_prefix(Some("#")),
        )
        .try_into_reader_with_file_path(Some(path.as_ref().to_path_buf()))
        .map_err(polars_err)?
        .finish()
        .map_err(polars_err)
}

fn str_col<'a>(df: &'a DataFrame, name: &str) -> Result<&'a StringChunked> {
    df.column(name)
        .map_err(polars_err)?
        .str()
        .map_err(polars_err)
}

/// Load GENIE `data_mutations_extended` rows.
pub fn load_genie_mutations(path: impl AsRef<std::path::Path>) -> Result<Vec<MutationRow>> {
    let df = read_table(path)?;
    let sample = str_col(&df, "Tumor_Sample_Barcode")?;
    let gene = str_col(&df, "Hugo_Symbol")?;
    let chrom = df
        .column("Chromosome")
        .map_err(polars_err)?
        .cast(&DataType::String)
        .map_err(polars_err)?;
    let chrom = chrom.str().map_err(polars_err)?;
    let pos = df
        .column("Start_Position")
        .map_err(polars_err)?
        .cast(&DataType::Int64)
        .map_err(polars_err)?;
    let pos = pos.i64().map_err(polars_err)?;
    let ref_a = str_col(&df, "Reference_Allele")?;
    let alt_a = str_col(&df, "Tumor_Seq_Allele2")?;

    let mut rows = Vec::with_capacity(df.height());
    for i in 0..df.height() {
        let (Some(s), Some(g)) = (sample.get(i), gene.get(i)) else {
            continue;
        };
        rows.push(MutationRow {
            sample_id: s.to_string(),
            gene: g.to_string(),
            chromosome: chrom.get(i).unwrap_or("").to_string(),
            position: pos.get(i).unwrap_or(0).max(0) as u64,
            reference_allele: ref_a.get(i).unwrap_or("").to_string(),
            alternate_allele: alt_a.get(i).unwrap_or("").to_string(),
        });
    }
    Ok(rows)
}

/// Derive (sample_id, gene) events for gene-count features (all mutations).
pub fn to_mutation_events(rows: &[MutationRow]) -> Vec<(String, String)> {
    rows.iter()
        .map(|r| (r.sample_id.clone(), r.gene.clone()))
        .collect()
}

/// Derive single-base-substitution calls (for SBS96). GENIE chromosomes are
/// prefixed with `chr` to match the genome reference convention.
pub fn to_snv_calls(rows: &[MutationRow]) -> Vec<SnvCall> {
    fn is_single_base(s: &str) -> bool {
        s.len() == 1
            && matches!(
                s.as_bytes()[0].to_ascii_uppercase(),
                b'A' | b'C' | b'G' | b'T'
            )
    }
    rows.iter()
        .filter(|r| is_single_base(&r.reference_allele) && is_single_base(&r.alternate_allele))
        .map(|r| SnvCall {
            sample_id: r.sample_id.clone(),
            chromosome: if r.chromosome.starts_with("chr") {
                r.chromosome.clone()
            } else {
                format!("chr{}", r.chromosome)
            },
            position: r.position,
            reference_allele: r.reference_allele.as_bytes()[0],
            alternate_allele: r.alternate_allele.as_bytes()[0],
        })
        .collect()
}

/// Load GENIE clinical patient + sample files and produce per-sample clinical
/// covariates. `AGE_AT_SEQ_REPORT` values may be strings like `">89"`.
pub fn load_genie_clinical(
    patient_path: impl AsRef<std::path::Path>,
    sample_path: impl AsRef<std::path::Path>,
) -> Result<Vec<PatientClinical>> {
    let patients = read_table(patient_path)?;
    let samples = read_table(sample_path)?;

    // Map PATIENT_ID -> SEX.
    let p_id = str_col(&patients, "PATIENT_ID")?;
    let p_sex = str_col(&patients, "SEX")?;
    let mut sex_of: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for i in 0..patients.height() {
        if let (Some(id), Some(sex)) = (p_id.get(i), p_sex.get(i)) {
            let code = match sex.trim().to_ascii_lowercase().as_str() {
                "male" => 1.0,
                "female" => -1.0,
                _ => 0.0,
            };
            sex_of.insert(id.to_string(), code);
        }
    }

    let s_sample = str_col(&samples, "SAMPLE_ID")?;
    let s_patient = str_col(&samples, "PATIENT_ID")?;
    let s_age = samples
        .column("AGE_AT_SEQ_REPORT")
        .map_err(polars_err)?
        .cast(&DataType::String)
        .map_err(polars_err)?;
    let s_age = s_age.str().map_err(polars_err)?;

    let mut out = Vec::with_capacity(samples.height());
    for i in 0..samples.height() {
        let Some(sample_id) = s_sample.get(i) else {
            continue;
        };
        let patient_id = s_patient.get(i).unwrap_or("");
        let sex_code = sex_of.get(patient_id).copied().unwrap_or(0.0);
        let age = parse_age(s_age.get(i).unwrap_or(""));
        out.push(PatientClinical {
            sample_id: sample_id.to_string(),
            sex_code,
            age,
        });
    }
    Ok(out)
}

/// Parse an age cell, handling `">89"`-style values (drop leading non-digit).
fn parse_age(raw: &str) -> f64 {
    let cleaned: String = raw.chars().filter(|c| c.is_ascii_digit()).collect();
    cleaned.parse::<f64>().unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn loads_mutations_and_derives_snvs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("muts.txt");
        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "# comment line").unwrap();
        writeln!(
            f,
            "Tumor_Sample_Barcode\tHugo_Symbol\tChromosome\tStart_Position\tReference_Allele\tTumor_Seq_Allele2"
        )
        .unwrap();
        writeln!(f, "GENIE-1\tERBB2\t17\t37868208\tC\tT").unwrap();
        writeln!(f, "GENIE-1\tTP53\t17\t7577121\tG\t-").unwrap(); // indel
        drop(f);

        let rows = load_genie_mutations(&path).unwrap();
        assert_eq!(rows.len(), 2);
        let events = to_mutation_events(&rows);
        assert_eq!(events.len(), 2);
        let snvs = to_snv_calls(&rows);
        assert_eq!(snvs.len(), 1); // indel filtered out
        assert_eq!(snvs[0].chromosome, "chr17");
        assert_eq!(snvs[0].reference_allele, b'C');
    }

    #[test]
    fn parses_capped_age() {
        assert_eq!(parse_age(">89"), 89.0);
        assert_eq!(parse_age("57"), 57.0);
        assert_eq!(parse_age("Unknown"), 0.0);
    }
}
