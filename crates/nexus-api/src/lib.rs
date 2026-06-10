//! # nexus-api
//!
//! Axum implementation of the Open Nexus prediction service contract
//! (`/api/v1/auth/register`, `/predictions/predict`, `/predictions/history`,
//! `/health`). Storage and feature building are pluggable so the service runs
//! with no external dependencies by default.

pub mod auth;
pub mod error;
pub mod handlers;
pub mod state;
pub mod storage;

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;

pub use state::{AppState, FeatureSource, RawFeatureSource};
pub use storage::{InMemoryStorage, Storage};

/// Build the application router for a given state.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/api/v1/auth/register", post(handlers::register))
        .route("/api/v1/predictions/predict", post(handlers::predict))
        .route("/api/v1/predictions/history", get(handlers::history))
        .route("/api/v1/health", get(handlers::health))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use base64::Engine;
    use std::sync::Arc;
    use tower::ServiceExt;

    fn auth_header(user: &str, pass: &str) -> String {
        let c = base64::engine::general_purpose::STANDARD.encode(format!("{user}:{pass}"));
        format!("Basic {c}")
    }

    #[tokio::test]
    async fn register_then_health_flow() {
        let state = AppState::new(Arc::new(InMemoryStorage::new()));
        let app = build_router(state);

        // Register.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/auth/register")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"username":"alice","password":"secret"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Health with valid creds.
        let resp = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/v1/health")
                    .header("authorization", auth_header("alice", "secret"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        // Health with bad creds.
        let resp = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/api/v1/health")
                    .header("authorization", auth_header("alice", "wrong"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn predict_without_model_is_unavailable() {
        let state = AppState::new(Arc::new(InMemoryStorage::new()));
        let app = build_router(state);
        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/auth/register")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"username":"bob","password":"secret"}"#))
                    .unwrap(),
            )
            .await
            .unwrap();

        let resp = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/v1/predictions/predict")
                    .header("authorization", auth_header("bob", "secret"))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"age":45,"gender":"male","cna_events":"","mutations":""}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
    }
}
