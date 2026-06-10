//! Loaders for the JSON artifacts emitted by `tools/pickle-migrate`
//! (replacing the upstream pickles): canonical feature manifest, model
//! metadata, and cohort age stats.

use serde::Deserialize;

use nexus_core::{error::Result, CohortAgeStats, FeatureManifest, ModelMetadata};

/// Load [`ModelMetadata`] from a JSON file (`{"features": [...], "target_classes": [...]}`).
pub fn load_model_metadata(path: impl AsRef<std::path::Path>) -> Result<ModelMetadata> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

/// Load the canonical [`FeatureManifest`].
///
/// Accepts either `{"features": [...]}` or a bare JSON array `[...]` (the form
/// produced by pickling a plain Python list).
pub fn load_feature_manifest(path: impl AsRef<std::path::Path>) -> Result<FeatureManifest> {
    let text = std::fs::read_to_string(path)?;

    #[derive(Deserialize)]
    #[serde(untagged)]
    enum ManifestRepr {
        Wrapped { features: Vec<String> },
        Bare(Vec<String>),
    }

    let repr: ManifestRepr = serde_json::from_str(&text)?;
    let features = match repr {
        ManifestRepr::Wrapped { features } => features,
        ManifestRepr::Bare(v) => v,
    };
    Ok(FeatureManifest::new(features))
}

/// Load [`CohortAgeStats`] (`{"Age_mean": .., "Std_mean": ..}`).
pub fn load_cohort_age_stats(path: impl AsRef<std::path::Path>) -> Result<CohortAgeStats> {
    let text = std::fs::read_to_string(path)?;
    Ok(serde_json::from_str(&text)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp(name: &str, content: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("nexus-artifacts-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join(name);
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        path
    }

    #[test]
    fn loads_manifest_both_shapes() {
        let p1 = write_temp("m1.json", r#"{"features":["A","B"]}"#);
        let p2 = write_temp("m2.json", r#"["A","B"]"#);
        assert_eq!(load_feature_manifest(&p1).unwrap().features, vec!["A", "B"]);
        assert_eq!(load_feature_manifest(&p2).unwrap().features, vec!["A", "B"]);
    }

    #[test]
    fn loads_age_stats() {
        let p = write_temp("age.json", r#"{"Age_mean":60.0,"Std_mean":12.0}"#);
        let s = load_cohort_age_stats(&p).unwrap();
        assert_eq!(s.age_mean, 60.0);
        assert_eq!(s.std_mean, 12.0);
    }

    #[test]
    fn loads_metadata() {
        let p = write_temp(
            "meta.json",
            r#"{"features":["Age","Sex"],"target_classes":["NSCLC","CRC"]}"#,
        );
        let m = load_model_metadata(&p).unwrap();
        assert_eq!(m.target_classes, vec!["NSCLC", "CRC"]);
    }
}
