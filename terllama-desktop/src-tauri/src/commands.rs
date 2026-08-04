use crate::config::Settings;
use crate::convert::{self, ConvertConfig};
use crate::download;
use crate::server::ServerManager;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tauri::Emitter;
use tauri::Manager;
use tauri::State;
use tokio::io::{AsyncBufReadExt, BufReader};

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

/// Parse a percentage value from a subprocess output line.
/// Recognises patterns like `42%`, `[download] 12.7%`, `5.2% of`.
fn parse_progress_pct(line: &str) -> Option<u64> {
    let bytes = line.as_bytes();
    for i in 0..bytes.len().saturating_sub(1) {
        if bytes[i] == b'%' {
            let mut j = i;
            while j > 0 && (bytes[j - 1].is_ascii_digit() || bytes[j - 1] == b'.') {
                j -= 1;
            }
            if j < i {
                if let Ok(num_str) = std::str::from_utf8(&bytes[j..i]) {
                    if let Ok(val) = num_str.parse::<f64>() {
                        if (0.0..=100.0).contains(&val) {
                            return Some(val as u64);
                        }
                    }
                }
            }
        }
    }
    None
}

#[tauri::command]
/// Download a model in one of three formats: "fp", "q4", or "ternary".
pub async fn download_model(
    app: tauri::AppHandle,
    model_id: String,
    format: String,
) -> Result<(), String> {
    // Determine HF repo from registry
    let registry = download::fetch_registry().await?;
    let model = registry
        .models
        .iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("Model {} not found in registry", model_id))?;

    let out_dir = download::get_models_dir().join(&model_id);
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("Failed to create dir: {}", e))?;

    match format.as_str() {
        // Original weights (safetensors) — direct multi-file download.
        "fp" => {
            let info = &model.formats.fp;
            if !info.available {
                return Err(format!("FP download unavailable: {}", info.note));
            }
            return download_files_direct(
                &app,
                &model.id,
                if info.hf_repo.is_empty() { &model.hf_repo } else { &info.hf_repo },
                if info.files.is_empty() { vec![info.filename.clone()] } else { info.files.clone() },
                info.size_mb,
                &out_dir,
            ).await;
        }
        // GGUF Q4_K_M — direct single-file download.
        "q4" => {
            let info = &model.formats.q4;
            if !info.available {
                return Err(format!("Q4 download unavailable: {}", info.note));
            }
            let repo = if info.hf_repo.is_empty() { &model.hf_repo } else { &info.hf_repo };
            return download_files_direct(
                &app,
                &model.id,
                repo,
                vec![info.filename.clone()],
                info.size_mb,
                &out_dir,
            ).await;
        }
        // Ternary — convert locally from FP weights unless pre-made weights exist.
        "ternary" => {
            let info = &model.formats.ternary;
            if !info.available {
                return Err(format!("Ternary unavailable: {}", info.note));
            }
            if !info.needs_conversion {
                // Pre-made ternary weights: direct download.
                let repo = if info.hf_repo.is_empty() { &model.hf_repo } else { &info.hf_repo };
                return download_files_direct(
                    &app,
                    &model.id,
                    repo,
                    vec![info.filename.clone()],
                    info.size_mb,
                    &out_dir,
                ).await;
            }
            // Conversion path (fall through to the python export flow below).
        }
        _ => return Err(format!("Unknown download format: {}", format)),
    }

    // Ternary format (als/i2s): run C++ binary with Python export script
    let resource_dir = app
        .path()
        .resource_dir()
        .ok();
    let binary = crate::server::find_terllama_binary(resource_dir.as_deref())?;

    // Resolve scripts directory (Tauri resource dir for bundled, or dev path)
    let scripts_dir = match crate::convert::get_scripts_dir(&app) {
        Ok(d) => d,
        Err(_) => {
            // Fallback: relative to CWD (dev mode)
            std::path::PathBuf::from("scripts")
        }
    };

    // Spawn subprocess with piped stdout/stderr so we can stream progress
    let mut child = tokio::process::Command::new(&binary)
        .arg("pull")
        .arg(&model.hf_repo)
        .arg("--fmt")
        .arg(&model.format)
        .arg("--outdir")
        .arg(&out_dir)
        .env("TERLLAMA_SCRIPT_DIR", scripts_dir.to_string_lossy().to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Failed to start download: {}", e))?;

    let stdout = child.stdout.take().ok_or("No stdout")?;
    let stderr = child.stderr.take().ok_or("No stderr")?;

    let last_pct = Arc::new(AtomicU64::new(0));

    // Stream stdout lines for progress
    let mid_so = model_id.clone();
    let app_so = app.clone();
    let lp_so = last_pct.clone();
    let so_handle = tokio::spawn(async move {
        let reader = BufReader::new(stdout);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if let Some(pct) = parse_progress_pct(&line) {
                let pct = pct.min(99);
                if pct > lp_so.load(Ordering::Relaxed) {
                    lp_so.store(pct, Ordering::Relaxed);
                    let _ = app_so.emit(
                        "download-progress",
                        download::DownloadProgressEvent {
                            model_id: mid_so.clone(),
                            file: "model.bin".to_string(),
                            downloaded: pct,
                            total: 100,
                            speed: 0.0,
                        },
                    );
                }
            }
        }
    });

    // Stream stderr lines for progress (some tools write progress there)
    // Also capture the last N lines in case the subprocess fails
    let mid_se = model_id.clone();
    let app_se = app.clone();
    let lp_se = last_pct.clone();
    let stderr_log = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
    let stderr_log_tx = stderr_log.clone();
    let se_handle = tokio::spawn(async move {
        let reader = BufReader::new(stderr);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            // Keep last 10 lines for error diagnostics
            {
                let mut log = stderr_log_tx.lock().unwrap();
                log.push(line.clone());
                if log.len() > 10 { log.remove(0); }
            }
            if let Some(pct) = parse_progress_pct(&line) {
                let pct = pct.min(99);
                if pct > lp_se.load(Ordering::Relaxed) {
                    lp_se.store(pct, Ordering::Relaxed);
                    let _ = app_se.emit(
                        "download-progress",
                        download::DownloadProgressEvent {
                            model_id: mid_se.clone(),
                            file: "model.bin".to_string(),
                            downloaded: pct,
                            total: 100,
                            speed: 0.0,
                        },
                    );
                }
            }
        }
    });

    // Wait for subprocess to finish
    let status = child.wait().await.map_err(|e| format!("Process error: {}", e))?;

    // Drain streamers
    let _ = so_handle.await;
    let _ = se_handle.await;

    if status.success() {
        // Emit final 100 % event
        let _ = app.emit(
            "download-progress",
            download::DownloadProgressEvent {
                model_id,
                file: "model.bin".to_string(),
                downloaded: 100,
                total: 100,
                speed: 0.0,
            },
        );
        Ok(())
    } else {
        // Include captured stderr in error message for diagnostics
        let stderr_lines = stderr_log.lock().unwrap();
        let detail: String = if stderr_lines.is_empty() {
            "unknown error (no stderr output)".to_string()
        } else {
            stderr_lines.join(" | ")
        };
        let code = status.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".to_string());
        Err(format!("Download failed (exit {}) — {}", code, detail))
    }
}

/// Download one or more files directly from HuggingFace (no Python conversion).
/// Emits `download-progress` events per file. `total_size_mb` is a fallback for
/// progress when the server doesn't send a Content-Length.
async fn download_files_direct(
    app: &tauri::AppHandle,
    model_id: &str,
    hf_repo: &str,
    filenames: Vec<String>,
    total_size_mb: u64,
    out_dir: &std::path::Path,
) -> Result<(), String> {
    let fallback_total = total_size_mb * 1024 * 1024;
    let client = reqwest::Client::builder()
        .user_agent("Terllama-Desktop/1.0.0")
        .build()
        .map_err(|e| format!("Failed to create client: {}", e))?;

    for filename in &filenames {
        let url = format!(
            "https://huggingface.co/{}/resolve/main/{}",
            hf_repo, filename
        );

        let _ = app.emit("download-progress", download::DownloadProgressEvent {
            model_id: model_id.to_string(),
            file: filename.clone(),
            downloaded: 0,
            total: fallback_total,
            speed: 0.0,
        });

        let resp = client
            .get(&url)
            .send()
            .await
            .map_err(|e| format!("Failed to download {}: {}", url, e))?;

        if !resp.status().is_success() {
            return Err(format!(
                "HTTP {} — file not found at {}",
                resp.status(),
                url
            ));
        }

        let total = resp.content_length().unwrap_or(fallback_total);
        let file_path = out_dir.join(filename);

        let mut file = tokio::fs::File::create(&file_path)
            .await
            .map_err(|e| format!("Failed to create file: {}", e))?;

        let mut stream = resp.bytes_stream();
        let mut downloaded: u64 = 0;
        let start = std::time::Instant::now();

        use futures_util::StreamExt;
        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Download error: {}", e))?;
            downloaded += chunk.len() as u64;

            tokio::io::AsyncWriteExt::write_all(&mut file, &chunk)
                .await
                .map_err(|e| format!("Write error: {}", e))?;

            let elapsed = start.elapsed().as_secs_f64();
            let speed = if elapsed > 0.0 { downloaded as f64 / elapsed } else { 0.0 };

            let pct = if total > 0 {
                (downloaded as f64 / total as f64 * 100.0) as u64
            } else {
                0
            };

            let _ = app.emit("download-progress", download::DownloadProgressEvent {
                model_id: model_id.to_string(),
                file: filename.clone(),
                downloaded: pct.min(99),
                total: fallback_total,
                speed,
            });
        }

        let _ = app.emit("download-progress", download::DownloadProgressEvent {
            model_id: model_id.to_string(),
            file: filename.clone(),
            downloaded: 100,
            total: fallback_total,
            speed: 0.0,
        });
    }

    Ok(())
}

#[tauri::command]
pub async fn list_downloaded_models() -> Result<Vec<download::DownloadedModel>, String> {
    let mut models = download::list_downloaded_models()?;

    // Map legacy slug-named dirs (e.g. "HuggingFaceTB-SmolLM2-135M" from the old
    // "/" → "-" naming) back to their registry id so they show as installed.
    if let Ok(registry) = download::fetch_registry().await {
        let mut slug_to_id: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        for m in &registry.models {
            slug_to_id.insert(m.hf_repo.replace('/', "-"), m.id.clone());
            slug_to_id.insert(m.id.clone(), m.id.clone());
        }
        for model in models.iter_mut() {
            if let Some(real_id) = slug_to_id.get(&model.id) {
                model.id = real_id.clone();
            }
        }
    }

    // De-duplicate: if both a slug dir and a proper dir exist, prefer the proper dir.
    let mut seen: std::collections::HashMap<String, bool> = std::collections::HashMap::new();
    models.retain(|m| {
        if seen.contains_key(&m.id) {
            return false;
        }
        seen.insert(m.id.clone(), true);
        true
    });

    Ok(models)
}

#[tauri::command]
pub async fn delete_model(model_id: String) -> Result<(), String> {
    let models_dir = download::get_models_dir();
    let path = models_dir.join(&model_id);
    if path.exists() {
        return std::fs::remove_dir_all(&path).map_err(|e| format!("Failed to delete: {}", e));
    }
    // Legacy slug-named dir (e.g. "HuggingFaceTB-SmolLM2-135M") — find it via registry.
    if let Ok(registry) = download::fetch_registry().await {
        for m in &registry.models {
            if m.id == model_id {
                let slug_path = models_dir.join(m.hf_repo.replace('/', "-"));
                if slug_path.exists() {
                    return std::fs::remove_dir_all(&slug_path)
                        .map_err(|e| format!("Failed to delete: {}", e));
                }
            }
        }
    }
    Err(format!("Model {} not found", model_id))
}

#[tauri::command]
pub async fn start_server(
    app: tauri::AppHandle,
    state: State<'_, AppState>,
    model_id: String,
    port: u16,
) -> Result<(), String> {
    // Refuse to load a model while a conversion is running — the weights are
    // being rewritten under the model dir and loading would race with it.
    if crate::convert::is_converting() {
        return Err(
            "Cannot load a model while a conversion is in progress. Wait for it to finish.".to_string(),
        );
    }
    let resource_dir = app.path().resource_dir().ok();
    state.server.start(model_id, port, resource_dir.as_deref()).await
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

/// Shared cancellation flag for conversion
use std::sync::atomic::AtomicBool;
use std::sync::OnceLock;
use tokio::sync::Mutex;

static CONVERT_CANCEL: OnceLock<Arc<AtomicBool>> = OnceLock::new();
static CONVERT_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

fn get_cancel_flag() -> Arc<AtomicBool> {
    CONVERT_CANCEL.get_or_init(|| Arc::new(AtomicBool::new(false))).clone()
}

#[tauri::command]
pub async fn convert_model(
    app: tauri::AppHandle,
    model: String,
    format: String,
    terms: u32,
) -> Result<(), String> {
    // Ensure only one conversion at a time
    let lock = CONVERT_LOCK.get_or_init(|| Mutex::new(()));
    let _lock = lock.lock().await;

    let cancel = get_cancel_flag();
    cancel.store(false, Ordering::SeqCst);

    let out_dir = dirs::home_dir()
        .unwrap_or_default()
        .join(".terllama/models")
        .join(model.replace('/', "-"))
        .to_string_lossy()
        .to_string();

    let config = ConvertConfig {
        model,
        format,
        terms,
        out_dir,
    };

    convert::run_conversion(app, cancel, config).await
}

#[tauri::command]
pub async fn cancel_conversion() -> Result<(), String> {
    get_cancel_flag().store(true, Ordering::SeqCst);
    Ok(())
}

#[tauri::command]
pub async fn check_python() -> Result<String, String> {
    let python = convert::find_python().ok_or_else(|| "Python 3 not found".to_string())?;
    let (major, minor) = convert::check_python_version(&python)?;
    Ok(format!("Python {}.{} at {}", major, minor, python.display()))
}

#[tauri::command]
pub async fn check_convert_deps() -> Result<bool, String> {
    let python = convert::find_python().ok_or_else(|| "Python 3 not found".to_string())?;
    let out = std::process::Command::new(&python)
        .args(["-c", "import torch, transformers; print('ok')"])
        .output()
        .map_err(|e| format!("Failed to check deps: {}", e))?;
    Ok(out.status.success())
}

#[tauri::command]
pub async fn check_update() -> Result<UpdateInfo, String> {
    let current_version = env!("CARGO_PKG_VERSION");
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
