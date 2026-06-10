//! Conversions between the core [`FeatureMatrix`] and Polars `DataFrame`,
//! plus Parquet / TSV persistence. The first column is always `sample_id`.

use polars::prelude::*;

use nexus_core::{error::Result, FeatureMatrix, NexusError};

const SAMPLE_ID_COL: &str = "sample_id";

fn polars_err(e: PolarsError) -> NexusError {
    NexusError::Invariant(format!("polars: {e}"))
}

/// Convert a feature matrix into a Polars `DataFrame`.
pub fn matrix_to_frame(m: &FeatureMatrix) -> Result<DataFrame> {
    let mut columns: Vec<Column> = Vec::with_capacity(m.n_features() + 1);
    columns.push(Column::new(SAMPLE_ID_COL.into(), m.sample_ids.clone()));
    for (j, name) in m.feature_names.iter().enumerate() {
        let col: Vec<f64> = (0..m.n_samples()).map(|i| m.values[[i, j]]).collect();
        columns.push(Column::new(name.as_str().into(), col));
    }
    DataFrame::new(columns).map_err(polars_err)
}

/// Convert a Polars `DataFrame` (with a `sample_id` column) into a feature
/// matrix. All non-`sample_id` columns are read as `f64`.
pub fn frame_to_matrix(df: &DataFrame) -> Result<FeatureMatrix> {
    let sample_ids: Vec<String> = df
        .column(SAMPLE_ID_COL)
        .map_err(polars_err)?
        .str()
        .map_err(polars_err)?
        .into_iter()
        .map(|o| o.unwrap_or("").to_string())
        .collect();

    let feature_names: Vec<String> = df
        .get_column_names()
        .into_iter()
        .filter(|c| c.as_str() != SAMPLE_ID_COL)
        .map(|c| c.to_string())
        .collect();

    let n = sample_ids.len();
    let mut values = ndarray::Array2::<f64>::zeros((n, feature_names.len()));
    for (j, name) in feature_names.iter().enumerate() {
        let s = df.column(name).map_err(polars_err)?;
        let ca = s.cast(&DataType::Float64).map_err(polars_err)?;
        let ca = ca.f64().map_err(polars_err)?;
        for (i, v) in ca.into_iter().enumerate() {
            values[[i, j]] = v.unwrap_or(0.0);
        }
    }
    FeatureMatrix::new(sample_ids, feature_names, values)
}

/// Write a feature matrix to Parquet.
pub fn write_parquet(m: &FeatureMatrix, path: impl AsRef<std::path::Path>) -> Result<()> {
    let mut df = matrix_to_frame(m)?;
    let file = std::fs::File::create(path)?;
    ParquetWriter::new(file)
        .finish(&mut df)
        .map_err(polars_err)?;
    Ok(())
}

/// Read a feature matrix from Parquet.
pub fn read_parquet(path: impl AsRef<std::path::Path>) -> Result<FeatureMatrix> {
    let file = std::fs::File::open(path)?;
    let df = ParquetReader::new(file).finish().map_err(polars_err)?;
    frame_to_matrix(&df)
}

/// Write a feature matrix to a tab-separated file.
pub fn write_tsv(m: &FeatureMatrix, path: impl AsRef<std::path::Path>) -> Result<()> {
    let mut df = matrix_to_frame(m)?;
    let file = std::fs::File::create(path)?;
    CsvWriter::new(file)
        .with_separator(b'\t')
        .include_header(true)
        .finish(&mut df)
        .map_err(polars_err)?;
    Ok(())
}

/// Read a tab-separated file into a Polars `DataFrame`.
pub fn read_tsv(path: impl AsRef<std::path::Path>) -> Result<DataFrame> {
    let pathbuf = path.as_ref().to_path_buf();
    CsvReadOptions::default()
        .with_has_header(true)
        .with_parse_options(CsvParseOptions::default().with_separator(b'\t'))
        .try_into_reader_with_file_path(Some(pathbuf))
        .map_err(polars_err)?
        .finish()
        .map_err(polars_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn sample_matrix() -> FeatureMatrix {
        FeatureMatrix::new(
            vec!["s1".into(), "s2".into()],
            vec!["ERBB2".into(), "Age".into()],
            array![[2.0, 0.5], [0.0, -0.3]],
        )
        .unwrap()
    }

    #[test]
    fn frame_roundtrip() {
        let m = sample_matrix();
        let df = matrix_to_frame(&m).unwrap();
        let back = frame_to_matrix(&df).unwrap();
        assert_eq!(back.sample_ids, m.sample_ids);
        assert_eq!(back.feature_names, m.feature_names);
        assert_eq!(back.values, m.values);
    }

    #[test]
    fn parquet_roundtrip() {
        let m = sample_matrix();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("features.parquet");
        write_parquet(&m, &path).unwrap();
        let back = read_parquet(&path).unwrap();
        assert_eq!(back.values, m.values);
        assert_eq!(back.feature_names, m.feature_names);
    }

    #[test]
    fn tsv_write_then_read_frame() {
        let m = sample_matrix();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("features.tsv");
        write_tsv(&m, &path).unwrap();
        let df = read_tsv(&path).unwrap();
        let back = frame_to_matrix(&df).unwrap();
        assert_eq!(back.values, m.values);
    }
}
