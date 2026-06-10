//! HTTP Basic authentication + Argon2 password hashing.

use argon2::password_hash::{
    rand_core::OsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString,
};
use argon2::Argon2;
use axum::http::HeaderMap;
use base64::Engine;

use crate::error::ApiError;

/// Hash a plaintext password with Argon2id.
pub fn hash_password(password: &str) -> Result<String, ApiError> {
    let salt = SaltString::generate(&mut OsRng);
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| ApiError::Internal(format!("hash error: {e}")))
}

/// Verify a plaintext password against a stored Argon2 hash.
pub fn verify_password(password: &str, stored_hash: &str) -> bool {
    match PasswordHash::new(stored_hash) {
        Ok(parsed) => Argon2::default()
            .verify_password(password.as_bytes(), &parsed)
            .is_ok(),
        Err(_) => false,
    }
}

/// Extract `(username, password)` from an HTTP Basic `Authorization` header.
pub fn parse_basic_auth(headers: &HeaderMap) -> Result<(String, String), ApiError> {
    let value = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .ok_or(ApiError::Unauthorized)?;
    let encoded = value.strip_prefix("Basic ").ok_or(ApiError::Unauthorized)?;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .map_err(|_| ApiError::Unauthorized)?;
    let decoded = String::from_utf8(decoded).map_err(|_| ApiError::Unauthorized)?;
    let (user, pass) = decoded.split_once(':').ok_or(ApiError::Unauthorized)?;
    Ok((user.to_string(), pass.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_then_verify() {
        let h = hash_password("secret_password").unwrap();
        assert!(verify_password("secret_password", &h));
        assert!(!verify_password("wrong", &h));
    }

    #[test]
    fn parses_basic_header() {
        let mut headers = HeaderMap::new();
        let cred = base64::engine::general_purpose::STANDARD.encode("alice:pw");
        headers.insert(
            axum::http::header::AUTHORIZATION,
            format!("Basic {cred}").parse().unwrap(),
        );
        let (u, p) = parse_basic_auth(&headers).unwrap();
        assert_eq!(u, "alice");
        assert_eq!(p, "pw");
    }
}
