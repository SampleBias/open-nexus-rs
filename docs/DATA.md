# Data & Model Artifacts

Open Nexus does not bundle clinical data or trained models. This document lists
what you need and how the formats map from the upstream project.

## Required artifacts

| Artifact | Format | Source | Produced by |
|----------|--------|--------|-------------|
| XGBoost model | `model.json` | `Booster.save_model(...)` | upstream training |
| Model metadata | `{"features": [...], "target_classes": [...]}` | `model_metadata.json` | upstream / `pickle-migrate` |
| Feature manifest | `{"features": [...]}` | `features_onconpc.pkl` | `pickle-migrate` |
| Cohort age stats | `{"Age_mean": .., "Std_mean": ..}` | `combined_cohort_age_stats.pkl` | `pickle-migrate` |
| Signature weights | COSMIC SBS CSVs (`Type`, `Subtype`, `*_GRCh38/37`) | SigProfiler / COSMIC | external |
| Reference genome | FASTA | GRCh37/38 | external |

## GENIE input files (for `process-features`)

- `data_mutations_extended_*.txt` — columns `Tumor_Sample_Barcode`, `Hugo_Symbol`,
  `Chromosome`, `Start_Position`, `Reference_Allele`, `Tumor_Seq_Allele2`.
- `data_clinical_patient_*.txt` — `PATIENT_ID`, `SEX`.
- `data_clinical_sample_*.txt` — `SAMPLE_ID`, `PATIENT_ID`, `AGE_AT_SEQ_REPORT`
  (handles capped `">89"` values).

## One-time pickle migration

```bash
python tools/pickle-migrate/migrate.py \
  --features-pkl  data/features_onconpc.pkl \
  --age-stats-pkl data/combined_cohort_age_stats.pkl \
  --metadata      models/trained_models/model_metadata.json \
  --out-dir       artifacts/
```

## Feature naming (canonical)

- somatic mutation gene: `ERBB2`
- copy-number alteration: `ERBB2 CNA`
- mutation signature: `SBS1`
- clinical: `Age`, `Sex` (Male=1, Female=-1)

The feature manifest defines the exact column order; every sample is zero-padded
and aligned to it before inference.
