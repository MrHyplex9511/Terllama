use crate::config::Settings;
use crate::download;
use crate::server::ServerManager;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tauri::Emitter;
use tauri::State;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

pub struct AppState {
    pub server: Arc<ServerManager>,
}

#[tauri::command]
pub async fn fetch_registry() -> Result<download::Registry, String> {
    download::fetch_registry().await
}

#[tauri::command]
pub async fn download_model(
    app: tauri::AppHandle,
    model_id: String,
    _quant: String,
) -> Result<(), String> {
    let binary = crate::server::find_terllama_binary()?;

    // Determine HF repo from registry
    let registry = download::fetch_registry().await?;
    let model = registry
        .models
        .iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("Model {} not found in registry", model_id))?;

    // Download using terllama binary
    let output = tokio::process::Command::new(&binary)
        .arg("pull")
        .arg(&model.hf_repo)
        .arg("--fmt")
        .arg(&model.format)
        .output()
        .await
        .map_err(|e| format!("Download failed: {}", e))?;

    if output.status.success() {
        let _ = app.emit(
            "download-progress",
            download::DownloadProgressEvent {
                model_id,
                file: "model.bin".to_string(),
                downloaded: 1,
                total: 1,
                speed: 0.0,
            },
        );
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).to_string())
    }
}

#[tauri::command]
pub async fn list_downloaded_models() -> Result<Vec<download::DownloadedModel>, String> {
    download::list_downloaded_models()
}

#[tauri::command]
pub async fn delete_model(model_id: String) -> Result<(), String> {
    let path = download::get_models_dir().join(&model_id);
    if path.exists() {
        std::fs::remove_dir_all(&path).map_err(|e| format!("Failed to delete: {}", e))
    } else {
        Err(format!("Model {} not found", model_id))
    }
}

#[tauri::command]
pub async fn start_server(
    state: State<'_, AppState>,
    model_id: String,
    port: u16,
) -> Result<(), String> {
    state.server.start(model_id, port).await
}

#[tauri::command]
pub async fn stop_server(state: State<'_, AppState>) -> Result<(), String> {
    state.server.stop()
}

#[tauri::command]
pub async fn server_status(
    state: State<'_, AppState>,
) -> Result<crate::server::ServerStatus, String> {
    Ok(state.server.get_status())
}

#[tauri::command]
pub async fn send_chat_message(messages: Vec<Message>, port: u16) -> Result<String, String> {
    let client = Client::new();
    let body = serde_json::json!({
        "model": "default",
        "messages": messages,
        "stream": false
    });
    let resp = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", port))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Chat request failed: {}", e))?;
    resp.text()
        .await
        .map_err(|e| format!("Failed to read response: {}", e))
}

#[tauri::command]
pub async fn stream_chat(
    app: tauri::AppHandle,
    messages: Vec<Message>,
    port: u16,
) -> Result<(), String> {
    let client = Client::new();
    let body = serde_json::json!({
        "model": "default",
        "messages": messages,
        "stream": true
    });
    let resp = client
        .post(format!("http://127.0.0.1:{}/v1/chat/completions", port))
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Stream request failed: {}", e))?;

    let mut stream = resp.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Stream error: {}", e))?;
        let text = String::from_utf8_lossy(&chunk);
        for line in text.lines() {
            if line.starts_with("data: ") {
                let data = line.trim_start_matches("data: ");
                if data == "[DONE]" {
                    let _ = app.emit("chat-done", "");
                } else {
                    let _ = app.emit("chat-token", data);
                }
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpdateInfo {
    pub update_available: bool,
    pub latest_version: String,
    pub current_version: String,
    pub download_url: String,
}

#[tauri::command]
pub async fn get_settings() -> Result<Settings, String> {
    Ok(Settings::load())
}

#[tauri::command]
pub async fn save_settings(settings: Settings) -> Result<(), String> {
    settings.save()
}

#[tauri::command]
pub async fn check_update() -> Result<UpdateInfo, String> {
    let current_version = "1.0.0";
    let client = Client::builder()
        .user_agent("Terllama-Desktop/1.0.0")
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {}", e))?;

    let resp = client
        .get("https://api.github.com/repos/MrHyplex9511/Terllama/releases/latest")
        .send()
        .await
        .map_err(|e| format!("Failed to check for updates: {}", e))?;

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("Failed to parse update response: {}", e))?;

    let latest_tag = body["tag_name"]
        .as_str()
        .unwrap_or("v1.0.0")
        .trim_start_matches('v')
        .to_string();

    let download_url = body["html_url"]
        .as_str()
        .unwrap_or("https://github.com/MrHyplex9511/Terllama/releases")
        .to_string();

    let update_available = latest_tag != current_version;

    Ok(UpdateInfo {
        update_available,
        latest_version: latest_tag,
        current_version: current_version.to_string(),
        download_url,
    })
}
