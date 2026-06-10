//! Cross-validation support: stratified k-fold splitting and per-fold age
//! normalization. Ports the splitting logic and age-standardization behaviour
//! from `codes/utils_training.py`.

use nexus_core::{error::Result, FeatureMatrix, NexusError};

/// A tiny deterministic PRNG (xorshift64*) so fold shuffling is reproducible
/// without pulling an RNG dependency.
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

/// Assign each sample to one of `k` folds, stratified by class label.
///
/// Within each class the samples are shuffled (seeded) then dealt
/// round-robin into folds, which keeps class proportions balanced across
/// folds — the intent of the Python `get_cancer_to_num_val_samples` logic.
pub fn stratified_kfold(labels: &[usize], k: usize, seed: u64) -> Result<Vec<usize>> {
    if k < 2 {
        return Err(NexusError::invariant("k_fold must be >= 2"));
    }
    let n = labels.len();
    let n_classes = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);

    let mut per_class: Vec<Vec<usize>> = vec![Vec::new(); n_classes];
    for (idx, &c) in labels.iter().enumerate() {
        per_class[c].push(idx);
    }

    let mut rng = XorShift64::new(seed);
    let mut fold_of = vec![0usize; n];
    for class_members in per_class.iter_mut() {
        rng.shuffle(class_members);
        for (i, &sample_idx) in class_members.iter().enumerate() {
            fold_of[sample_idx] = i % k;
        }
    }
    Ok(fold_of)
}

/// Age standardization parameters fit on a training fold.
#[derive(Debug, Clone, Copy)]
pub struct AgeNorm {
    pub mean: f64,
    pub std: f64,
}

impl AgeNorm {
    pub fn transform(&self, age: f64) -> f64 {
        if self.std == 0.0 {
            0.0
        } else {
            (age - self.mean) / self.std
        }
    }
}

/// Fit age mean/std over the given training rows of the `Age` column.
pub fn fit_age_norm(matrix: &FeatureMatrix, train_rows: &[usize]) -> Result<AgeNorm> {
    let age_col = matrix
        .feature_index("Age")
        .ok_or_else(|| NexusError::feature_mismatch("no 'Age' column for normalization"))?;
    if train_rows.is_empty() {
        return Err(NexusError::invariant("empty training rows for age norm"));
    }
    let vals: Vec<f64> = train_rows
        .iter()
        .map(|&r| matrix.values[[r, age_col]])
        .collect();
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
    Ok(AgeNorm {
        mean,
        std: var.sqrt(),
    })
}

/// Return a copy of `matrix` with the `Age` column standardized in place
/// using the provided parameters.
pub fn apply_age_norm(matrix: &FeatureMatrix, norm: AgeNorm) -> Result<FeatureMatrix> {
    let age_col = matrix
        .feature_index("Age")
        .ok_or_else(|| NexusError::feature_mismatch("no 'Age' column for normalization"))?;
    let mut values = matrix.values.clone();
    for r in 0..matrix.n_samples() {
        values[[r, age_col]] = norm.transform(values[[r, age_col]]);
    }
    FeatureMatrix::new(
        matrix.sample_ids.clone(),
        matrix.feature_names.clone(),
        values,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn folds_are_balanced_and_complete() {
        let labels = vec![0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        let folds = stratified_kfold(&labels, 2, 42).unwrap();
        assert_eq!(folds.len(), labels.len());
        for &f in &folds {
            assert!(f < 2);
        }
        // class 0 (4 samples) split 2/2 across folds
        let c0_fold0 = labels
            .iter()
            .zip(&folds)
            .filter(|(&l, &f)| l == 0 && f == 0)
            .count();
        assert_eq!(c0_fold0, 2);
    }

    #[test]
    fn age_norm_zero_centers_training() {
        let m = FeatureMatrix::new(
            vec!["a".into(), "b".into(), "c".into()],
            vec!["Age".into(), "GENEA".into()],
            array![[40.0, 1.0], [50.0, 0.0], [60.0, 1.0]],
        )
        .unwrap();
        let norm = fit_age_norm(&m, &[0, 1, 2]).unwrap();
        assert!((norm.mean - 50.0).abs() < 1e-9);
        let normed = apply_age_norm(&m, norm).unwrap();
        let age_col = normed.feature_index("Age").unwrap();
        let mean: f64 = (0..3).map(|r| normed.values[[r, age_col]]).sum::<f64>() / 3.0;
        assert!(mean.abs() < 1e-9);
    }
}
