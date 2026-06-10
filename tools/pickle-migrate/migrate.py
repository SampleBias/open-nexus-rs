#!/usr/bin/env python3
"""One-time migration of upstream OncoNPC pickle artifacts to JSON.

This is the ONLY place Python is needed in the Open Nexus (Rust) project, and
it is run once (not in CI). It converts the pickled artifacts shipped with
`itmoon7/onconpc` into the JSON schemas consumed by the Rust crates.

Usage:
    python migrate.py \
        --features-pkl  data/features_onconpc.pkl \
        --age-stats-pkl data/combined_cohort_age_stats.pkl \
        --metadata      models/trained_models/model_metadata.json \
        --out-dir       artifacts/

Outputs:
    artifacts/features_onconpc.json        {"features": [...]}
    artifacts/combined_cohort_age_stats.json {"Age_mean": .., "Std_mean": ..}
    artifacts/model_metadata.json          (copied / validated)
"""
import argparse
import json
import os
import pickle


def load_pickle(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features-pkl")
    ap.add_argument("--age-stats-pkl")
    ap.add_argument("--metadata")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.features_pkl:
        features = list(load_pickle(args.features_pkl))
        with open(os.path.join(args.out_dir, "features_onconpc.json"), "w") as fp:
            json.dump({"features": features}, fp, indent=2)
        print(f"wrote features_onconpc.json ({len(features)} features)")

    if args.age_stats_pkl:
        stats = load_pickle(args.age_stats_pkl)
        # Expect keys Age_mean / Std_mean.
        out = {"Age_mean": float(stats["Age_mean"]), "Std_mean": float(stats["Std_mean"])}
        with open(os.path.join(args.out_dir, "combined_cohort_age_stats.json"), "w") as fp:
            json.dump(out, fp, indent=2)
        print("wrote combined_cohort_age_stats.json")

    if args.metadata:
        with open(args.metadata) as fp:
            meta = json.load(fp)
        assert "features" in meta and "target_classes" in meta, "metadata schema mismatch"
        with open(os.path.join(args.out_dir, "model_metadata.json"), "w") as fp:
            json.dump(meta, fp, indent=2)
        print("wrote model_metadata.json")


if __name__ == "__main__":
    main()
