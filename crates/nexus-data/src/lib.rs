//! # nexus-data
//!
//! Polars-backed I/O for Open Nexus. This is both the production data layer and
//! the substrate for `nexus-testkit`'s snapshot comparisons.
//!
//! - [`frame`]: [`nexus_core::FeatureMatrix`] <-> Polars `DataFrame`, Parquet & TSV.
//! - [`artifacts`]: JSON loaders for the migrated pickles.
//! - [`signatures_io`]: signature weight CSV loading.
//! - [`genie`]: AACR GENIE mutation/clinical loaders.

pub mod artifacts;
pub mod frame;
pub mod genie;
pub mod signatures_io;

pub use artifacts::{load_cohort_age_stats, load_feature_manifest, load_model_metadata};
pub use frame::{
    frame_to_matrix, matrix_to_frame, read_parquet, read_tsv, write_parquet, write_tsv,
};
pub use genie::{
    load_genie_clinical, load_genie_mutations, to_mutation_events, to_snv_calls, MutationRow,
};
pub use signatures_io::{load_signature_csv, load_signature_dir};
