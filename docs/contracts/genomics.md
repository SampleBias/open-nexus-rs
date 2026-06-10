# Contract: nexus-genomics

Reference: `codes/utils.py` (`get_snv_in_trinuc_context`, `obtain_mutation_signatures`,
`pre_process_features_genie`) and `process_features.py`.

## Public surface

- `trait GenomeReference { fn base_at(&self, chrom: &str, pos: u64) -> Option<u8>; }`
- `fn build_sbs96<G: GenomeReference>(calls, genome) -> (samples, channels[96], counts)`
- `SignatureSet::project(channels, counts) -> (sig_names, matrix)`
- `assemble_feature_matrix(mutations, cna, sig_samples, sig_names, sig_matrix, clinical) -> FeatureMatrix`
- `RawFeatureBuilder::build(&[(sample_id, RawPatientInput)]) -> FeatureMatrix`
- `FastaGenome` (default `GenomeReference` impl)

## Invariants (property-tested)

- SBS96 output has exactly **96** channels, labelled `5'[ref>alt]3'`, pyrimidine-normalized.
- Purine references are reverse-complemented.
- `GL*` / `chrMT` contigs and non-ACGT alleles are excluded.
- Signature projection requires exactly 96 shared channels (matched by label).
- Samples with `Age == 0` are dropped during assembly.

## Parity gate

`tests/snapshots/sbs96_expected.parquet`, `features_expected.parquet` via
`nexus-testkit::assert_matrix_near`.
