//! Request handlers implementing the documented endpoints.

use axum::extract::State;
use axum::http::HeaderMap;
use axum::Json;
use serde::Deserialize;
use serde_json::{json, Value};

use nexus_core::RawPatientInput;

use crate::auth::{hash_password, parse_basic_auth, verify_password};
use crate::error::ApiError;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct RegisterRequest {
    username: String,
    password: String,
}

#[derive(Deserialize)]
pub struct PredictRequest {
    age: f64,
    gender: String,
    #[serde(default)]
    cna_events: String,
    #[serde(default)]
    mutations: String,
}

/// Authenticate the request, returning the username on success.
fn authenticate(state: &AppState, headers: &HeaderMap) -> Result<String, ApiError> {
    let (username, password) = parse_basic_auth(headers)?;
    let stored = state
        .storage
        .password_hash(&username)
        .ok_or(ApiError::Unauthorized)?;
    if verify_password(&password, &stored) {
        Ok(username)
    } else {
        Err(ApiError::Unauthorized)
    }
}

/// `POST /api/v1/auth/register`
pub async fn register(
    State(state): State<AppState>,
    Json(req): Json<RegisterRequest>,
) -> Result<Json<Value>, ApiError> {
    if req.username.is_empty() || req.password.len() < 4 {
        return Err(ApiError::BadRequest(
            "username required and password must be >= 4 chars".into(),
        ));
    }
    let hash = hash_password(&req.password)?;
    state.storage.register(&req.username, hash)?;
    Ok(Json(
        json!({ "message": "user registered", "username": req.username }),
    ))
}

/// `POST /api/v1/predictions/predict`
pub async fn predict(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<PredictRequest>,
) -> Result<Json<Value>, ApiError> {
    let username = authenticate(&state, &headers)?;
    state.storage.check_rate_limit(&username)?;

    let predictor = state
        .predictor
        .as_ref()
        .ok_or_else(|| ApiError::ServiceUnavailable("model not loaded".into()))?;
    let feature_source = state
        .feature_source
        .as_ref()
        .ok_or_else(|| ApiError::ServiceUnavailable("feature source not configured".into()))?;

    let input = RawPatientInput {
        age: req.age,
        gender: req.gender,
        cna_events: req.cna_events,
        mutations: req.mutations,
    };
    let features = feature_source.build(&[("0".to_string(), input)])?;
    let preds = predictor.predict_top_n(&features, 3)?;

    // Shape the response like the README example.
    let predictions: Vec<Value> = preds
        .iter()
        .enumerate()
        .map(|(i, sp)| {
            let inner: Vec<Value> = sp
                .predictions
                .iter()
                .map(|p| json!({"cancer_type": p.cancer_type, "probability": p.probability}))
                .collect();
            json!({ "sample_id": i, "predictions": inner })
        })
        .collect();
    let response = json!({ "predictions": predictions });

    state.storage.save_prediction(&username, response.clone());
    Ok(Json(response))
}

/// `GET /api/v1/predictions/history`
pub async fn history(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    let username = authenticate(&state, &headers)?;
    let records = state.storage.history(&username);
    Ok(Json(json!({ "predictions": records })))
}

/// `GET /api/v1/health`
pub async fn health(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Value>, ApiError> {
    authenticate(&state, &headers)?;
    Ok(Json(json!({
        "status": "ok",
        "model_loaded": state.model_loaded(),
    })))
}
