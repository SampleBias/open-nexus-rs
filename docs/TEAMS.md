# Development Teams & Subagent Orchestration

The workspace is partitioned so independent teams (each driven by a Cursor
subagent) can work in parallel behind stable crate contracts.

| Team | Owns | Contract surface |
|------|------|------------------|
| **Team 0 — Foundation** | `nexus-core`, `nexus-data`, `tools/pickle-migrate` | Domain types, `FeatureMatrix`, JSON artifact schemas |
| **Team 1 — Genomics** | `nexus-genomics` | `GenomeReference`, `build_sbs96`, `SignatureSet::project`, `assemble_feature_matrix` |
| **Team 2 — ML** | `nexus-ml` | `TreeEnsemble`, `Predictor`, CV/filtering/metrics |
| **Team 3 — SHAP/Viz** | `nexus-shap`, `nexus-viz` | `ensemble_shap`, `Explainer`, `render_explanation_svg` |
| **Team 4 — API** | `nexus-api` | Axum routes, `Storage`, `FeatureSource` traits |
| **Team 5 — Desktop** | `nexus-desktop` | Tauri commands `predict`, `explain` |
| **Team 6 — CLI** | `nexus-cli` | `nexus` subcommands |
| **Team 7 — QA/Integration** | `nexus-testkit`, `tests/snapshots/`, CI | Polars snapshot diffs, property tests |

## Working agreement

1. **Contract-first.** Team 0 publishes/changes shared types in `nexus-core`;
   downstream teams depend only on those.
2. **Branch-per-team** (`team/1-genomics`, ...), merged via Team 7 integration PRs.
3. **Parquet snapshots** are the single source of parity truth (exported once
   from Python; never regenerated in CI).
4. **Green gate**: `cargo fmt --check`, `cargo clippy -D warnings`, `cargo test`.

## Subagent prompt skeleton

```
You own crates/<crate>. Implement <capability> to parity with the Python
reference in <file/function>. Read docs/contracts/<crate>.md.
Do not modify other crates' public APIs. Pass nexus-testkit snapshot/property
tests. Keep CI green (fmt + clippy -D warnings + test).
```
