# Contract: nexus-ml & nexus-shap

Reference: `codes/utils.py` (`get_xgboost_latest_cancer_type_preds`,
`obtain_shap_values`, `get_top_n_pred_and_shap`) and `utils_training.py`.

## nexus-ml surface

- `TreeEnsemble::from_json_slice / from_json_path` — parse XGBoost `save_model` JSON.
- `TreeEnsemble::predict_margins / predict_proba_row` — margins then softmax.
- `Predictor::predict / predict_top_n` — aligns to `ModelMetadata.features`, returns
  ranked `SamplePrediction`s.
- `low_frequency_filter`, `stratified_kfold`, `fit_age_norm`/`apply_age_norm`,
  `classification_report`, `filter_by_posterior_cutoff`.

### Inference parity

Probabilities equal upstream `predict(output_margin=True)` + softmax. `base_score`
cancels under softmax for `multi:softprob`; retained for SHAP base values.

## nexus-shap surface

- `tree_shap`, `ensemble_shap` — exact polynomial-time Tree SHAP.
- `Explainer::explain_sample / explain_all` — top-N grouped explanation for the
  predicted class.

### Invariant (property-tested)

**Local accuracy**: for every class `c`, `sum_i phi[c][i] + base[c] == margin_c(x)`.

## Parity gate

`tests/snapshots/predictions_expected.parquet`, `shap_expected.parquet`.
