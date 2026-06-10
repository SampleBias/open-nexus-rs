//! A simple in-memory FASTA-backed [`GenomeReference`].
//!
//! Loads each contig's sequence into memory keyed by the first token of the
//! header line. Suitable for exome panels / test references; for a full genome
//! prefer a future memory-mapped `.fai` implementation (the trait makes that a
//! drop-in replacement).

use std::collections::HashMap;

use nexus_core::error::Result;

use crate::sbs96::GenomeReference;

/// FASTA sequences indexed by contig name.
#[derive(Debug, Default)]
pub struct FastaGenome {
    contigs: HashMap<String, Vec<u8>>,
}

impl FastaGenome {
    /// Parse FASTA text into an in-memory genome.
    pub fn parse(text: &str) -> Self {
        let mut contigs = HashMap::new();
        let mut current: Option<(String, Vec<u8>)> = None;
        for line in text.lines() {
            if let Some(stripped) = line.strip_prefix('>') {
                if let Some((name, seq)) = current.take() {
                    contigs.insert(name, seq);
                }
                let name = stripped.split_whitespace().next().unwrap_or("").to_string();
                current = Some((name, Vec::new()));
            } else if let Some((_, seq)) = current.as_mut() {
                seq.extend(line.trim().bytes().map(|b| b.to_ascii_uppercase()));
            }
        }
        if let Some((name, seq)) = current.take() {
            contigs.insert(name, seq);
        }
        Self { contigs }
    }

    /// Load a FASTA file from disk.
    pub fn from_path(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(Self::parse(&text))
    }

    pub fn contig_count(&self) -> usize {
        self.contigs.len()
    }
}

impl GenomeReference for FastaGenome {
    fn base_at(&self, chromosome: &str, position: u64) -> Option<u8> {
        let seq = self.contigs.get(chromosome)?;
        if position == 0 {
            return None;
        }
        seq.get((position - 1) as usize).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_bases_one_based() {
        let fa = FastaGenome::parse(">chr1 desc\nACGTACGT\n>chr2\nTTTT\n");
        assert_eq!(fa.contig_count(), 2);
        assert_eq!(fa.base_at("chr1", 1), Some(b'A'));
        assert_eq!(fa.base_at("chr1", 4), Some(b'T'));
        assert_eq!(fa.base_at("chr2", 2), Some(b'T'));
        assert_eq!(fa.base_at("chr1", 0), None);
        assert_eq!(fa.base_at("chrX", 1), None);
    }
}
