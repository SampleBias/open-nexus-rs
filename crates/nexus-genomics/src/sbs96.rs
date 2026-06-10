//! SBS96 single-base-substitution trinucleotide context counting.
//!
//! Replaces the `SigProfilerMatrixGenerator` dependency used by
//! `codes/utils.py::get_snv_in_trinuc_context`. The 96 channels are the six
//! pyrimidine-normalized substitution types (`C>A, C>G, C>T, T>A, T>C, T>G`)
//! crossed with the 16 flanking trinucleotide contexts, labelled in the
//! standard `5'[ref>alt]3'` form (e.g. `"A[C>A]A"`).
//!
//! Determining the flanking bases requires a reference genome. Rather than
//! bundling a multi-gigabyte FASTA, we depend on a small [`GenomeReference`]
//! trait so callers can plug in a FASTA-backed (or test) provider. SNVs on
//! `GL*` / `chrMT` contigs are skipped, matching the Python filter.

use std::collections::BTreeMap;

use ndarray::Array2;

use nexus_core::error::{NexusError, Result};

/// Supplies reference genome bases for trinucleotide context lookup.
pub trait GenomeReference {
    /// Return the uppercase base (`A`/`C`/`G`/`T`) at 1-based `position` on
    /// `chromosome`, or `None` if unavailable (out of range / unknown contig).
    fn base_at(&self, chromosome: &str, position: u64) -> Option<u8>;
}

/// A minimal SNV record needed for SBS96 classification.
#[derive(Debug, Clone)]
pub struct SnvCall {
    pub sample_id: String,
    pub chromosome: String,
    pub position: u64,
    pub reference_allele: u8,
    pub alternate_allele: u8,
}

#[inline]
fn complement(base: u8) -> u8 {
    match base.to_ascii_uppercase() {
        b'A' => b'T',
        b'T' => b'A',
        b'C' => b'G',
        b'G' => b'C',
        other => other,
    }
}

#[inline]
fn is_acgt(base: u8) -> bool {
    matches!(base.to_ascii_uppercase(), b'A' | b'C' | b'G' | b'T')
}

/// Build the canonical ordered list of all 96 SBS channel labels.
pub fn sbs96_channels() -> Vec<String> {
    let subs = [
        (b'C', b'A'),
        (b'C', b'G'),
        (b'C', b'T'),
        (b'T', b'A'),
        (b'T', b'C'),
        (b'T', b'G'),
    ];
    let bases = [b'A', b'C', b'G', b'T'];
    let mut out = Vec::with_capacity(96);
    for (r, a) in subs {
        for &five in &bases {
            for &three in &bases {
                out.push(format!(
                    "{}[{}>{}]{}",
                    five as char, r as char, a as char, three as char
                ));
            }
        }
    }
    out
}

/// Compute the pyrimidine-normalized SBS96 channel label for one SNV, given
/// its flanking bases. Returns `None` for non-ACGT alleles.
pub fn classify_channel(five: u8, reference: u8, alt: u8, three: u8) -> Option<String> {
    if !(is_acgt(five) && is_acgt(reference) && is_acgt(alt) && is_acgt(three)) {
        return None;
    }
    let (five, reference, alt, three) = match reference.to_ascii_uppercase() {
        b'C' | b'T' => (
            five.to_ascii_uppercase(),
            reference.to_ascii_uppercase(),
            alt.to_ascii_uppercase(),
            three.to_ascii_uppercase(),
        ),
        // Purine reference: reverse-complement so the mutated base is a pyrimidine.
        _ => (
            complement(three),
            complement(reference),
            complement(alt),
            complement(five),
        ),
    };
    Some(format!(
        "{}[{}>{}]{}",
        five as char, reference as char, alt as char, three as char
    ))
}

/// Should this contig be skipped? Mirrors the Python `~contains('GL|chrMT')`.
pub fn is_skipped_contig(chrom: &str) -> bool {
    chrom.contains("GL") || chrom.contains("chrMT") || chrom.contains("MT")
}

/// Count SBS96 channels per sample.
///
/// Returns the row sample IDs (sorted, unique) and an `[n_samples x 96]`
/// count matrix whose columns are ordered by [`sbs96_channels`].
pub fn build_sbs96<G: GenomeReference>(
    calls: &[SnvCall],
    genome: &G,
) -> Result<(Vec<String>, Vec<String>, Array2<f64>)> {
    let channels = sbs96_channels();
    let channel_index: BTreeMap<&str, usize> = channels
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i))
        .collect();

    // Stable, sorted sample ordering.
    let mut sample_ids: Vec<String> = calls.iter().map(|c| c.sample_id.clone()).collect();
    sample_ids.sort();
    sample_ids.dedup();
    let sample_row: BTreeMap<&str, usize> = sample_ids
        .iter()
        .enumerate()
        .map(|(i, s)| (s.as_str(), i))
        .collect();

    let mut counts = Array2::<f64>::zeros((sample_ids.len(), channels.len()));

    for call in calls {
        if is_skipped_contig(&call.chromosome) {
            continue;
        }
        if !(is_acgt(call.reference_allele) && is_acgt(call.alternate_allele)) {
            continue; // indels / multi-nucleotide variants are not SNVs
        }
        let five = match genome.base_at(&call.chromosome, call.position.saturating_sub(1)) {
            Some(b) => b,
            None => continue,
        };
        let three = match genome.base_at(&call.chromosome, call.position + 1) {
            Some(b) => b,
            None => continue,
        };
        if let Some(channel) =
            classify_channel(five, call.reference_allele, call.alternate_allele, three)
        {
            let row = sample_row[call.sample_id.as_str()];
            let col = *channel_index
                .get(channel.as_str())
                .ok_or_else(|| NexusError::invariant(format!("unknown channel {channel}")))?;
            counts[[row, col]] += 1.0;
        }
    }

    Ok((sample_ids, channels, counts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// In-memory genome: maps (chrom, pos) -> base.
    struct MockGenome(HashMap<(String, u64), u8>);
    impl GenomeReference for MockGenome {
        fn base_at(&self, chromosome: &str, position: u64) -> Option<u8> {
            self.0.get(&(chromosome.to_string(), position)).copied()
        }
    }

    #[test]
    fn channels_are_96_unique() {
        let ch = sbs96_channels();
        assert_eq!(ch.len(), 96);
        let set: std::collections::BTreeSet<_> = ch.iter().collect();
        assert_eq!(set.len(), 96);
        assert_eq!(ch[0], "A[C>A]A");
    }

    #[test]
    fn purine_reference_is_reverse_complemented() {
        // G>T at context 5'=A, 3'=C  -> revcomp -> C>A with 5'=G(comp of C), 3'=T(comp of A)
        let label = classify_channel(b'A', b'G', b'T', b'C').unwrap();
        assert_eq!(label, "G[C>A]T");
    }

    #[test]
    fn counts_one_snv() {
        // chr1 pos 100: ref C alt A; flanks pos99='A', pos101='A' -> "A[C>A]A"
        let mut g = HashMap::new();
        g.insert(("chr1".to_string(), 99), b'A');
        g.insert(("chr1".to_string(), 101), b'A');
        let genome = MockGenome(g);

        let calls = vec![SnvCall {
            sample_id: "s1".into(),
            chromosome: "chr1".into(),
            position: 100,
            reference_allele: b'C',
            alternate_allele: b'A',
        }];
        let (samples, channels, counts) = build_sbs96(&calls, &genome).unwrap();
        assert_eq!(samples, vec!["s1"]);
        let idx = channels.iter().position(|c| c == "A[C>A]A").unwrap();
        assert_eq!(counts[[0, idx]], 1.0);
        assert_eq!(counts.sum(), 1.0);
    }

    #[test]
    fn skips_indels_and_excluded_contigs() {
        let genome = MockGenome(HashMap::new());
        let calls = vec![
            SnvCall {
                sample_id: "s1".into(),
                chromosome: "chrMT".into(),
                position: 5,
                reference_allele: b'C',
                alternate_allele: b'A',
            },
            SnvCall {
                sample_id: "s1".into(),
                chromosome: "chr1".into(),
                position: 5,
                reference_allele: b'C',
                alternate_allele: b'-', // indel
            },
        ];
        let (_s, _c, counts) = build_sbs96(&calls, &genome).unwrap();
        assert_eq!(counts.sum(), 0.0);
    }
}
