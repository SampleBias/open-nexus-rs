//! Pluggable persistence + rate limiting.
//!
//! The default [`InMemoryStorage`] keeps everything in process memory, so the
//! service builds and runs with zero external dependencies. A MongoDB-backed
//! implementation (matching the README's `mongodb://localhost:27017/open_nexus`)
//! is a drop-in follow-up behind this same trait.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::error::ApiError;

/// Limits from the README: 200 requests/day, 50 requests/hour.
const MAX_PER_DAY: usize = 200;
const MAX_PER_HOUR: usize = 50;

/// Persistence + rate-limiting backend.
pub trait Storage: Send + Sync {
    /// Register a new user. Returns `BadRequest` if the username exists.
    fn register(&self, username: &str, password_hash: String) -> Result<(), ApiError>;
    /// Stored Argon2 hash for a user, if any.
    fn password_hash(&self, username: &str) -> Option<String>;
    /// Append a prediction record to a user's history.
    fn save_prediction(&self, username: &str, record: serde_json::Value);
    /// All prediction records for a user (most recent last).
    fn history(&self, username: &str) -> Vec<serde_json::Value>;
    /// Record a request and enforce rate limits.
    fn check_rate_limit(&self, username: &str) -> Result<(), ApiError>;
}

#[derive(Default)]
struct Inner {
    users: HashMap<String, String>, // username -> password hash
    predictions: HashMap<String, Vec<serde_json::Value>>,
    requests: HashMap<String, Vec<Instant>>,
}

/// In-memory [`Storage`] implementation.
#[derive(Default)]
pub struct InMemoryStorage {
    inner: Mutex<Inner>,
}

impl InMemoryStorage {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Storage for InMemoryStorage {
    fn register(&self, username: &str, password_hash: String) -> Result<(), ApiError> {
        let mut g = self.inner.lock().unwrap();
        if g.users.contains_key(username) {
            return Err(ApiError::BadRequest("username already exists".into()));
        }
        g.users.insert(username.to_string(), password_hash);
        Ok(())
    }

    fn password_hash(&self, username: &str) -> Option<String> {
        self.inner.lock().unwrap().users.get(username).cloned()
    }

    fn save_prediction(&self, username: &str, record: serde_json::Value) {
        self.inner
            .lock()
            .unwrap()
            .predictions
            .entry(username.to_string())
            .or_default()
            .push(record);
    }

    fn history(&self, username: &str) -> Vec<serde_json::Value> {
        self.inner
            .lock()
            .unwrap()
            .predictions
            .get(username)
            .cloned()
            .unwrap_or_default()
    }

    fn check_rate_limit(&self, username: &str) -> Result<(), ApiError> {
        let mut g = self.inner.lock().unwrap();
        let now = Instant::now();
        let times = g.requests.entry(username.to_string()).or_default();
        times.retain(|t| now.duration_since(*t) < Duration::from_secs(86_400));
        let last_hour = times
            .iter()
            .filter(|t| now.duration_since(**t) < Duration::from_secs(3_600))
            .count();
        if times.len() >= MAX_PER_DAY || last_hour >= MAX_PER_HOUR {
            return Err(ApiError::RateLimited);
        }
        times.push(now);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_is_idempotent_guarded() {
        let s = InMemoryStorage::new();
        assert!(s.register("a", "h".into()).is_ok());
        assert!(s.register("a", "h".into()).is_err());
    }

    #[test]
    fn history_round_trips() {
        let s = InMemoryStorage::new();
        s.save_prediction("a", serde_json::json!({"x": 1}));
        assert_eq!(s.history("a").len(), 1);
        assert_eq!(s.history("b").len(), 0);
    }
}
