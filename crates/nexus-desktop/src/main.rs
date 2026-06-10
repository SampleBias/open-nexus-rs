//! Open Nexus desktop application (Tauri 2).

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::sync::Arc;

use nexus_core::RawPatientInput;
use nexus_desktop::Engine;

#[tauri::command]
fn predict(
    engine: tauri::State<Arc<Engine>>,
    input: RawPatientInput,
) -> Result<nexus_desktop::PredictResponse, String> {
    nexus_desktop::predict(engine.inner().clone(), input)
}

#[tauri::command]
fn explain(
    engine: tauri::State<Arc<Engine>>,
    input: RawPatientInput,
) -> Result<nexus_desktop::ExplainResponse, String> {
    nexus_desktop::explain(engine.inner().clone(), input)
}

fn main() {
    let engine = Arc::new(Engine::from_env().expect("failed to load model artifacts"));
    tauri::Builder::default()
        .manage(engine)
        .invoke_handler(tauri::generate_handler![predict, explain])
        .run(tauri::generate_context!())
        .expect("error while running Open Nexus desktop");
}
