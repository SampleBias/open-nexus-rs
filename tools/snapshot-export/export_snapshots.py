#!/usr/bin/env python3
"""One-time export of reference outputs from the upstream Python pipeline to
Parquet snapshots committed under `tests/snapshots/`.

Run ONCE (or when intentionally refreshing baselines). Never invoked from CI:
CI compares the Rust pipeline output against the committed Parquet using
`nexus-testkit` (Polars), with no Python at test time.

This script expects an importable upstream `onconpc` checkout (via PYTHONPATH).
It writes:
    tests/snapshots/features_expected.parquet
    tests/snapshots/predictions_expected.parquet
    tests/snapshots/shap_expected.parquet

The exact extraction depends on the upstream notebooks; fill in the marked
sections with calls into `codes/utils.py`.
"""
import argparse
import os

import pandas as pd  # noqa: F401  (used when wiring upstream calls)


def export(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # --- features_expected.parquet -----------------------------------------
    # features_df = pre_process_features_genie(...)   # from codes/utils.py
    # features_df.reset_index(names="sample_id").to_parquet(
    #     os.path.join(out_dir, "features_expected.parquet"))

    # --- predictions_expected.parquet --------------------------------------
    # preds_df = get_xgboost_latest_cancer_type_preds(model, features_df, cancer_types)
    # preds_df.reset_index(names="sample_id").to_parquet(
    #     os.path.join(out_dir, "predictions_expected.parquet"))

    # --- shap_expected.parquet ---------------------------------------------
    # shap_df = obtain_shap_values(model, features_df) ...
    # shap_df.to_parquet(os.path.join(out_dir, "shap_expected.parquet"))

    print(
        "Snapshot export scaffold complete. Wire the marked sections to the\n"
        "upstream onconpc functions, then commit the Parquet files under\n"
        f"{out_dir}. CI will diff Rust output against them via nexus-testkit."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="tests/snapshots")
    args = ap.parse_args()
    export(args.out_dir)
