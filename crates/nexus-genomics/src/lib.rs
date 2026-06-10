//! # nexus-genomics
//!
//! Genomic feature engineering for Open Nexus, replacing the
//! SigProfilerMatrixGenerator + pandas pipeline in `codes/utils.py`.
//!
//! - [`sbs96`]: trinucleotide-context SBS96 counting (with a pluggable
//!   [`sbs96::GenomeReference`]).
//! - [`signatures`]: linear projection of counts onto mutation signatures.
//! - [`pipeline`]: assembly of the full feature matrix and the raw-input
//!   inference path ([`pipeline::RawFeatureBuilder`]).

pub mod fasta;
pub mod pipeline;
pub mod sbs96;
pub mod signatures;

pub use fasta::FastaGenome;
pub use pipeline::{assemble_feature_matrix, CnaValue, PatientClinical, RawFeatureBuilder};
pub use sbs96::{build_sbs96, sbs96_channels, GenomeReference, SnvCall};
pub use signatures::{Signature, SignatureSet};
