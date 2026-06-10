//! Feature-matrix assembly.
//!
//! Ports the merging logic of `pre_process_features_genie` /
//! `pre_process_features_dfci` and the raw-input path
//! `get_onconpc_features_from_raw_data`. Produces a [`FeatureMatrix`] with
//! standardized feature names ready to be aligned to the canonical manifest.

use std::collections::{BTreeMap, BTreeSet};

use ndarray::Array2;

use nexus_core::{
    error::Result, CohortAgeStats, FeatureManifest, FeatureMatrix, MutationRecord, RawPatientInput,
};

use crate::sbs96::{build_sbs96, GenomeReference, SnvCall};
use crate::signatures::SignatureSet;

/// Clinical covariates for one sample.
#[derive(Debug, Clone)]
pub struct PatientClinical {
    pub sample_id: String,
    /// Encoded sex (Male=1, Female=-1, unknown=0).
    pub sex_code: f64,
    /// Raw age in years (standardized later).
    pub age: f64,
}

/// A copy-number alteration value for a (sample, gene) pair.
#[derive(Debug, Clone)]
pub struct CnaValue {
    pub sample_id: String,
    pub gene: String,
    pub value: f64,
}

/// Assemble a feature matrix from the four feature families.
///
/// Rows are based on the clinical cohort (left-joined with the other sources,
/// missing values filled with 0). Samples with `age == 0` are dropped, matching
/// the Python NaN-age handling.
pub fn assemble_feature_matrix(
    mutation_events: &[(String, String)], // (sample_id, gene)
    cna_values: &[CnaValue],
    signature_samples: &[String],
    signature_names: &[String],
    signature_matrix: &Array2<f64>,
    clinical: &[PatientClinical],
) -> Result<FeatureMatrix> {
    // Per-sample gene mutation counts.
    let mut mut_counts: BTreeMap<&str, BTreeMap<&str, f64>> = BTreeMap::new();
    let mut mutation_genes: BTreeSet<&str> = BTreeSet::new();
    for (sample, gene) in mutation_events {
        *mut_counts
            .entry(sample.as_str())
            .or_default()
            .entry(gene.as_str())
            .or_insert(0.0) += 1.0;
        mutation_genes.insert(gene.as_str());
    }

    // Per-sample CNA values.
    let mut cna_map: BTreeMap<&str, BTreeMap<&str, f64>> = BTreeMap::new();
    let mut cna_genes: BTreeSet<&str> = BTreeSet::new();
    for c in cna_values {
        cna_map
            .entry(c.sample_id.as_str())
            .or_default()
            .insert(c.gene.as_str(), c.value);
        cna_genes.insert(c.gene.as_str());
    }

    // Signature lookup by sample.
    let sig_row: BTreeMap<&str, usize> = signature_samples
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    // Final column order: mutation genes, CNA genes, signatures, Age, Sex.
    let mut feature_names: Vec<String> = Vec::new();
    feature_names.extend(mutation_genes.iter().map(|g| g.to_string()));
    feature_names.extend(cna_genes.iter().map(|g| format!("{g} CNA")));
    feature_names.extend(signature_names.iter().cloned());
    let age_col = feature_names.len();
    feature_names.push("Age".to_string());
    feature_names.push("Sex".to_string());

    // Rows: clinical samples with non-zero age.
    let kept: Vec<&PatientClinical> = clinical.iter().filter(|c| c.age != 0.0).collect();
    let sample_ids: Vec<String> = kept.iter().map(|c| c.sample_id.clone()).collect();

    let mut values = Array2::<f64>::zeros((kept.len(), feature_names.len()));
    for (row, patient) in kept.iter().enumerate() {
        let sid = patient.sample_id.as_str();
        let mut col = 0;
        if let Some(genes) = mut_counts.get(sid) {
            for g in &mutation_genes {
                if let Some(&v) = genes.get(g) {
                    values[[row, col]] = v;
                }
                col += 1;
            }
        } else {
            col += mutation_genes.len();
        }
        if let Some(genes) = cna_map.get(sid) {
            for g in &cna_genes {
                if let Some(&v) = genes.get(g) {
                    values[[row, col]] = v;
                }
                col += 1;
            }
        } else {
            col += cna_genes.len();
        }
        if let Some(&srow) = sig_row.get(sid) {
            for (k, _name) in signature_names.iter().enumerate() {
                values[[row, col + k]] = signature_matrix[[srow, k]];
            }
        }
        values[[row, age_col]] = patient.age;
        values[[row, age_col + 1]] = patient.sex_code;
    }

    FeatureMatrix::new(sample_ids, feature_names, values)
}

/// End-to-end builder for the raw-input inference path.
///
/// Mirrors `get_onconpc_features_from_raw_data`: build SBS96 from the patient's
/// SNVs, project signatures, assemble features, align to the canonical
/// manifest (zero-padding), then standardize `Age` with cohort stats.
pub struct RawFeatureBuilder<'a, G: GenomeReference> {
    pub genome: &'a G,
    pub signatures: &'a SignatureSet,
    pub manifest: &'a FeatureManifest,
    pub age_stats: CohortAgeStats,
}

impl<'a, G: GenomeReference> RawFeatureBuilder<'a, G> {
    pub fn build(&self, inputs: &[(String, RawPatientInput)]) -> Result<FeatureMatrix> {
        let mut snvs: Vec<SnvCall> = Vec::new();
        let mut mutation_events: Vec<(String, String)> = Vec::new();
        let mut cna_values: Vec<CnaValue> = Vec::new();
        let mut clinical: Vec<PatientClinical> = Vec::new();

        for (sample_id, input) in inputs {
            for m in input.parse_mutations()? {
                mutation_events.push((sample_id.clone(), m.gene.clone()));
                push_snv(&mut snvs, sample_id, &m);
            }
            for e in input.parse_cna()? {
                cna_values.push(CnaValue {
                    sample_id: sample_id.clone(),
                    gene: e.gene,
                    value: e.value as f64,
                });
            }
            clinical.push(PatientClinical {
                sample_id: sample_id.clone(),
                sex_code: input.sex_code(),
                age: input.age,
            });
        }

        let (sig_samples, channels, counts) = build_sbs96(&snvs, self.genome)?;
        let (sig_names, sig_matrix) = if sig_samples.is_empty() {
            (
                self.signatures.names(),
                Array2::<f64>::zeros((0, self.signatures.signatures.len())),
            )
        } else {
            self.signatures.project(&channels, &counts)?
        };

        let assembled = assemble_feature_matrix(
            &mutation_events,
            &cna_values,
            &sig_samples,
            &sig_names,
            &sig_matrix,
            &clinical,
        )?;

        // Align to canonical features then standardize Age.
        let mut aligned = assembled.align_to(&self.manifest.features)?;
        if let Some(age_col) = aligned.feature_index("Age") {
            for r in 0..aligned.n_samples() {
                let raw = aligned.values[[r, age_col]];
                aligned.values[[r, age_col]] = self.age_stats.standardize(raw);
            }
        }
        Ok(aligned)
    }
}

fn push_snv(snvs: &mut Vec<SnvCall>, sample_id: &str, m: &MutationRecord) {
    let ref_b = m
        .reference_allele
        .as_bytes()
        .first()
        .copied()
        .unwrap_or(b'N');
    let alt_b = m
        .alternate_allele
        .as_bytes()
        .first()
        .copied()
        .unwrap_or(b'N');
    // Only single-base substitutions contribute to SBS96.
    if m.reference_allele.len() == 1 && m.alternate_allele.len() == 1 {
        snvs.push(SnvCall {
            sample_id: sample_id.to_string(),
            chromosome: m.chromosome.clone(),
            position: m.position,
            reference_allele: ref_b,
            alternate_allele: alt_b,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assembles_basic_matrix() {
        let mutation_events = vec![
            ("s1".to_string(), "ERBB2".to_string()),
            ("s1".to_string(), "ERBB2".to_string()),
            ("s2".to_string(), "TP53".to_string()),
        ];
        let cna = vec![CnaValue {
            sample_id: "s1".into(),
            gene: "RAF1".into(),
            value: 2.0,
        }];
        let sig_samples = vec!["s1".to_string(), "s2".to_string()];
        let sig_names = vec!["SBS1".to_string()];
        let mut sig_mat = Array2::<f64>::zeros((2, 1));
        sig_mat[[0, 0]] = 0.7;
        sig_mat[[1, 0]] = 0.3;
        let clinical = vec![
            PatientClinical {
                sample_id: "s1".into(),
                sex_code: 1.0,
                age: 60.0,
            },
            PatientClinical {
                sample_id: "s2".into(),
                sex_code: -1.0,
                age: 50.0,
            },
        ];

        let fm = assemble_feature_matrix(
            &mutation_events,
            &cna,
            &sig_samples,
            &sig_names,
            &sig_mat,
            &clinical,
        )
        .unwrap();

        assert_eq!(fm.n_samples(), 2);
        // ERBB2 count for s1 should be 2.
        let erbb2 = fm.feature_index("ERBB2").unwrap();
        let s1 = fm.sample_ids.iter().position(|s| s == "s1").unwrap();
        assert_eq!(fm.values[[s1, erbb2]], 2.0);
        // RAF1 CNA present.
        assert!(fm.feature_index("RAF1 CNA").is_some());
        // SBS1 present.
        assert!(fm.feature_index("SBS1").is_some());
    }

    #[test]
    fn drops_zero_age_samples() {
        let clinical = vec![
            PatientClinical {
                sample_id: "s1".into(),
                sex_code: 1.0,
                age: 0.0,
            },
            PatientClinical {
                sample_id: "s2".into(),
                sex_code: -1.0,
                age: 50.0,
            },
        ];
        let fm =
            assemble_feature_matrix(&[], &[], &[], &[], &Array2::zeros((0, 0)), &clinical).unwrap();
        assert_eq!(fm.sample_ids, vec!["s2"]);
    }
}
