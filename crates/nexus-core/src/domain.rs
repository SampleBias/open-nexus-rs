//! User-facing prediction & explanation domain types, plus parsers for the
//! REST/CLI input string formats documented in the upstream README.

use serde::{Deserialize, Serialize};

use crate::error::{NexusError, Result};
use crate::feature::FeatureGroup;

/// A single cancer-type probability.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Prediction {
    pub cancer_type: String,
    pub probability: f64,
}

/// All predictions for one sample, sorted descending by probability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplePrediction {
    pub sample_id: String,
    /// Top cancer type (argmax).
    pub cancer_type: String,
    /// Probability of the argmax class.
    pub max_posterior: f64,
    /// Full ranked list (or top-N).
    pub predictions: Vec<Prediction>,
}

/// One feature's contribution to a prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapFeature {
    pub feature_name: String,
    pub group: FeatureGroup,
    /// SHAP attribution value (signed).
    pub shap_value: f64,
    /// The raw feature value for this sample.
    pub feature_value: f64,
}

/// A full SHAP explanation for one sample's predicted class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapExplanation {
    pub sample_id: String,
    pub predicted_cancer_type: String,
    pub predicted_probability: f64,
    /// Features sorted by descending |shap_value| (then truncated to top-N
    /// by callers as needed).
    pub features: Vec<ShapFeature>,
}

/// A single parsed somatic mutation (gene + genomic coordinate).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MutationRecord {
    pub gene: String,
    pub chromosome: String,
    pub position: u64,
    pub reference_allele: String,
    pub alternate_allele: String,
}

/// A single parsed copy-number alteration event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CnaEvent {
    pub gene: String,
    /// CNA level in `[-2, 2]`.
    pub value: i8,
}

/// Raw patient input as received from the API/CLI before feature assembly.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawPatientInput {
    pub age: f64,
    pub gender: String,
    /// e.g. `"RAF1 2 | PPARG 2"`
    pub cna_events: String,
    /// e.g. `"ERBB2, chr17, 37868208, C, T"` (multiple separated by `;`)
    pub mutations: String,
}

impl RawPatientInput {
    /// Encode gender to the model's numeric convention (Male = 1, Female = -1,
    /// unknown = 0), matching `pre_process_features_genie`.
    pub fn sex_code(&self) -> f64 {
        match self.gender.trim().to_ascii_lowercase().as_str() {
            "male" | "m" => 1.0,
            "female" | "f" => -1.0,
            _ => 0.0,
        }
    }

    /// Parse the CNA spec string into structured events.
    pub fn parse_cna(&self) -> Result<Vec<CnaEvent>> {
        parse_cna_events(&self.cna_events)
    }

    /// Parse the mutation spec string into structured records.
    pub fn parse_mutations(&self) -> Result<Vec<MutationRecord>> {
        parse_mutations(&self.mutations)
    }
}

/// Parse `"RAF1 2 | PPARG 2"` into CNA events.
pub fn parse_cna_events(spec: &str) -> Result<Vec<CnaEvent>> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for token in spec.split('|') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        let mut parts = token.split_whitespace();
        let gene = parts
            .next()
            .ok_or_else(|| NexusError::parse(format!("CNA token missing gene: '{token}'")))?;
        let value_str = parts
            .next()
            .ok_or_else(|| NexusError::parse(format!("CNA token missing value: '{token}'")))?;
        let value: i8 = value_str
            .parse()
            .map_err(|_| NexusError::parse(format!("invalid CNA value '{value_str}'")))?;
        if !(-2..=2).contains(&value) {
            return Err(NexusError::parse(format!(
                "CNA value {value} out of range [-2, 2]"
            )));
        }
        out.push(CnaEvent {
            gene: gene.to_string(),
            value,
        });
    }
    Ok(out)
}

/// Parse `"ERBB2, chr17, 37868208, C, T"` (or several, separated by `;`)
/// into mutation records.
pub fn parse_mutations(spec: &str) -> Result<Vec<MutationRecord>> {
    let spec = spec.trim();
    if spec.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for record in spec.split(';') {
        let record = record.trim();
        if record.is_empty() {
            continue;
        }
        let fields: Vec<&str> = record.split(',').map(|s| s.trim()).collect();
        if fields.len() != 5 {
            return Err(NexusError::parse(format!(
                "mutation must have 5 comma-separated fields (gene, chr, pos, ref, alt): '{record}'"
            )));
        }
        let position: u64 = fields[2]
            .parse()
            .map_err(|_| NexusError::parse(format!("invalid mutation position '{}'", fields[2])))?;
        out.push(MutationRecord {
            gene: fields[0].to_string(),
            chromosome: fields[1].to_string(),
            position,
            reference_allele: fields[3].to_string(),
            alternate_allele: fields[4].to_string(),
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_cna_readme_example() {
        let events = parse_cna_events("RAF1 2 | PPARG 2").unwrap();
        assert_eq!(
            events,
            vec![
                CnaEvent {
                    gene: "RAF1".into(),
                    value: 2
                },
                CnaEvent {
                    gene: "PPARG".into(),
                    value: 2
                },
            ]
        );
    }

    #[test]
    fn parse_mutation_readme_example() {
        let muts = parse_mutations("ERBB2, chr17, 37868208, C, T").unwrap();
        assert_eq!(
            muts,
            vec![MutationRecord {
                gene: "ERBB2".into(),
                chromosome: "chr17".into(),
                position: 37_868_208,
                reference_allele: "C".into(),
                alternate_allele: "T".into(),
            }]
        );
    }

    #[test]
    fn sex_code_maps_correctly() {
        let mk = |g: &str| RawPatientInput {
            age: 50.0,
            gender: g.into(),
            cna_events: String::new(),
            mutations: String::new(),
        };
        assert_eq!(mk("male").sex_code(), 1.0);
        assert_eq!(mk("Female").sex_code(), -1.0);
        assert_eq!(mk("other").sex_code(), 0.0);
    }

    #[test]
    fn cna_out_of_range_errors() {
        assert!(parse_cna_events("RAF1 9").is_err());
    }
}
