//! Mutation-signature projection.
//!
//! Ports `codes/utils.py::obtain_mutation_signatures`. Despite the name, the
//! reference pipeline does **not** perform NNLS deconvolution here (that lives
//! only in the experimental `Mutation_Analysis.py`); it computes a linear
//! projection of the 96-channel trinucleotide counts onto each signature's
//! reference weight vector:
//!
//! ```text
//! signature_value[sample, sig] = sum_channel counts[sample, channel] * weight[sig, channel]
//! ```
//!
//! Channels are matched by their string label (`"A[C>A]A"`), exactly like the
//! Python `set(mut_df.index) & set(df_trinuc_feats.columns)` intersection.

use std::collections::BTreeMap;

use ndarray::Array2;

use nexus_core::error::{NexusError, Result};

/// A single mutational signature: a name and per-channel weights.
#[derive(Debug, Clone)]
pub struct Signature {
    pub name: String,
    pub weights: BTreeMap<String, f64>,
}

impl Signature {
    /// Validate that the weights approximately sum to 1 (Python tolerance 0.1).
    pub fn validate(&self) -> Result<()> {
        let sum: f64 = self.weights.values().sum();
        if (sum - 1.0).abs() > 0.1 {
            return Err(NexusError::invariant(format!(
                "signature '{}' weights sum to {sum}, expected ~1.0",
                self.name
            )));
        }
        Ok(())
    }
}

/// A collection of signatures to project onto.
#[derive(Debug, Clone, Default)]
pub struct SignatureSet {
    pub signatures: Vec<Signature>,
}

impl SignatureSet {
    pub fn new(signatures: Vec<Signature>) -> Self {
        Self { signatures }
    }

    pub fn names(&self) -> Vec<String> {
        self.signatures.iter().map(|s| s.name.clone()).collect()
    }

    /// Project an SBS96 count matrix onto the signatures.
    ///
    /// `channels` labels the columns of `counts` (`[n_samples x n_channels]`).
    /// Returns the signature names and a `[n_samples x n_signatures]` matrix.
    pub fn project(
        &self,
        channels: &[String],
        counts: &Array2<f64>,
    ) -> Result<(Vec<String>, Array2<f64>)> {
        let n_samples = counts.nrows();
        let mut out = Array2::<f64>::zeros((n_samples, self.signatures.len()));

        for (sig_idx, sig) in self.signatures.iter().enumerate() {
            // Common channels between the count columns and this signature.
            let common: Vec<(usize, f64)> = channels
                .iter()
                .enumerate()
                .filter_map(|(col, ch)| sig.weights.get(ch).map(|&w| (col, w)))
                .collect();
            if common.len() != 96 {
                return Err(NexusError::invariant(format!(
                    "signature '{}' shares {} channels with counts, expected 96",
                    sig.name,
                    common.len()
                )));
            }
            for row in 0..n_samples {
                let mut acc = 0.0;
                for &(col, w) in &common {
                    acc += counts[[row, col]] * w;
                }
                out[[row, sig_idx]] = acc;
            }
        }
        Ok((self.names(), out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sbs96::sbs96_channels;

    fn uniform_signature(name: &str) -> Signature {
        let channels = sbs96_channels();
        let w = 1.0 / channels.len() as f64;
        Signature {
            name: name.to_string(),
            weights: channels.into_iter().map(|c| (c, w)).collect(),
        }
    }

    #[test]
    fn projects_counts_onto_uniform_signature() {
        let channels = sbs96_channels();
        let mut counts = Array2::<f64>::zeros((1, 96));
        counts[[0, 0]] = 3.0;
        counts[[0, 5]] = 1.0; // total 4 mutations

        let set = SignatureSet::new(vec![uniform_signature("SBS_TEST")]);
        let (names, proj) = set.project(&channels, &counts).unwrap();
        assert_eq!(names, vec!["SBS_TEST"]);
        // uniform weight * total counts = (1/96)*4
        assert!((proj[[0, 0]] - 4.0 / 96.0).abs() < 1e-12);
    }

    #[test]
    fn uniform_signature_validates() {
        assert!(uniform_signature("SBS1").validate().is_ok());
    }
}
