use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ServerStatus {
    Stopped,
    Starting,
    Running,
    Error(String),
}

pub struct ServerManager {
    pub status: Arc<Mutex<ServerStatus>>,
    pub process: Arc<Mutex<Option<Child>>>,
    pub current_model: Arc<Mutex<Option<String>>>,
    pub port: Arc<Mutex<u16>>,
}

impl ServerManager {
    pub fn new() -> Self {
        Self {
            status: Arc::new(Mutex::new(ServerStatus::Stopped)),
            process: Arc::new(Mutex::new(None)),
            current_model: Arc::new(Mutex::new(None)),
            port: Arc::new(Mutex::new(8375)),
        }
    }

    pub async fn start(
        &self,
        model_id: String,
        port: u16,
        resource_dir: Option<&std::path::Path>,
    ) -> Result<(), String> {
        {
            let mut status = self.status.lock().map_err(|e| e.to_string())?;
            *status = ServerStatus::Starting;
        }

        let models_dir = dirs::home_dir()
            .unwrap_or_default()
            .join(".terllama/models")
            .join(&model_id);

        // Resolve the HF repo name for this model (registry id -> hf_repo).
        // decode_helper.py needs the real repo id (e.g. "HuggingFaceTB/SmolLM2-135M"),
        // which differs from the registry id used as the directory name.
        let hf_repo = resolve_hf_repo(&model_id).await;

        let binary = find_terllama_binary(resource_dir)?;

        // Scripts dir for engine Python helpers: bundled resources first,
        // then dev layout relative to the binary's repo.
        let scripts_dir = resource_dir
            .map(|rd| rd.join("resources").join("scripts"))
            .filter(|p| p.exists())
            .or_else(|| {
                let dev = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("resources")
                    .join("scripts");
                dev.exists().then_some(dev)
            });

        // Log file: ~/.terllama/logs/server-<model_id>.log
        let log_dir = dirs::home_dir()
            .unwrap_or_default()
            .join(".terllama/logs");
        std::fs::create_dir_all(&log_dir).map_err(|e| format!("Failed to create log dir: {}", e))?;
        let log_path = log_dir.join(format!("server-{}.log", model_id));

        let mut cmd = Command::new(&binary);
        cmd.arg("serve")
            .arg("--port")
            .arg(port.to_string())
            .env(
                "TERLLAMA_MODEL_DIR",
                models_dir.to_string_lossy().to_string(),
            )
            .env("TERLLAMA_PORT", port.to_string())
            .env("TERLLAMA_HF_MODEL", hf_repo);
        if let Some(sd) = scripts_dir {
            cmd.env("TERLLAMA_SCRIPT_DIR", sd.to_string_lossy().to_string());
        }

        let mut child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to start server: {}", e))?;

        // Capture engine stdout/stderr: tee to log file + keep last 30 lines
        // for diagnostics if startup fails.
        let recent_lines: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let log_path_clone = log_path.clone();
        let recent_stdout = recent_lines.clone();
        let stdout = child.stdout.take();
        let stderr = child.stderr.take();
        let so_thread = std::thread::spawn(move || {
            if let Some(out) = stdout {
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&log_path_clone)
                    .ok();
                let reader = BufReader::new(out);
                for line in reader.lines() {
                    if let Ok(l) = line {
                        if let Some(f) = file.as_mut() {
                            let _ = writeln!(f, "{}", l);
                        }
                        let mut buf = recent_stdout.lock().unwrap();
                        buf.push(l);
                        if buf.len() > 30 {
                            buf.remove(0);
                        }
                    }
                }
            }
        });
        let log_path_clone2 = log_path.clone();
        let log_path_display = log_path;
        let recent_stderr = recent_lines.clone();
        let se_thread = std::thread::spawn(move || {
            if let Some(err) = stderr {
                let mut file = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&log_path_clone2)
                    .ok();
                let reader = BufReader::new(err);
                for line in reader.lines() {
                    if let Ok(l) = line {
                        if let Some(f) = file.as_mut() {
                            let _ = writeln!(f, "{}", l);
                        }
                        let mut buf = recent_stderr.lock().unwrap();
                        buf.push(l);
                        if buf.len() > 30 {
                            buf.remove(0);
                        }
                    }
                }
            }
        });

        // Poll health endpoint
        let health_url = format!("http://127.0.0.1:{}/health", port);
        let max_retries = 60u32;
        let mut started = false;

        for _ in 0..max_retries {
            tokio::time::sleep(Duration::from_millis(500)).await;

            // Check if process is still alive
            if let Ok(Some(status)) = child.try_wait() {
                // Engine exited before becoming healthy — collect its output.
                let _ = so_thread.join();
                let _ = se_thread.join();
                let lines = recent_lines.lock().unwrap().clone();
                let detail = if lines.is_empty() {
                    format!("engine exited early (status: {:?})", status.code())
                } else {
                    format!(
                        "engine exited early (status: {:?}). Last output:\n{}",
                        status.code(),
                        lines.join("\n")
                    )
                };
                *self.status.lock().map_err(|e| e.to_string())? =
                    ServerStatus::Error(format!(
                        "Server failed to start. {} Log file: {}",
                        detail,
                        log_path_display.display()
                    ));
                return Err(format!(
                    "Server failed to start within 30 seconds. {} Log file: {}",
                    detail,
                    log_path_display.display()
                ));
            }

            if let Ok(resp) = reqwest::get(&health_url).await {
                if resp.status().is_success() {
                    started = true;
                    break;
                }
            }
        }

        let mut status = self.status.lock().map_err(|e| e.to_string())?;
        if started {
            *status = ServerStatus::Running;
            *self.process.lock().map_err(|e| e.to_string())? = Some(child);
            *self.current_model.lock().map_err(|e| e.to_string())? = Some(model_id);
            *self.port.lock().map_err(|e| e.to_string())? = port;
            Ok(())
        } else {
            let _ = child.kill();
            let _ = child.wait();
            let _ = so_thread.join();
            let _ = se_thread.join();
            let lines = recent_lines.lock().unwrap().clone();
            let detail = if lines.is_empty() {
                "No output captured from engine.".to_string()
            } else {
                format!("Last engine output:\n{}", lines.join("\n"))
            };
            *status = ServerStatus::Error(format!(
                "Server failed to start within 30s. {} Log file: {}",
                detail,
                log_path_display.display()
            ));
            Err(format!(
                "Server failed to start within 30 seconds. {} Log file: {}",
                detail,
                log_path_display.display()
            ))
        }
    }

    pub fn stop(&self) -> Result<(), String> {
        if let Ok(mut proc) = self.process.lock() {
            if let Some(ref mut child) = *proc {
                let _ = child.kill();
                // Wait up to 5s
                for _ in 0..10 {
                    std::thread::sleep(Duration::from_millis(500));
                    if let Ok(Some(_)) = child.try_wait() {
                        break;
                    }
                }
                let _ = child.kill();
                let _ = child.wait();
            }
        }
        *self.process.lock().map_err(|e| e.to_string())? = None;
        *self.status.lock().map_err(|e| e.to_string())? = ServerStatus::Stopped;
        *self.current_model.lock().map_err(|e| e.to_string())? = None;
        Ok(())
    }

    pub fn get_status(&self) -> ServerStatus {
        self.status
            .lock()
            .map(|s| s.clone())
            .unwrap_or(ServerStatus::Error("Lock poisoned".to_string()))
    }
}

pub fn find_terllama_binary(resource_dir: Option<&std::path::Path>) -> Result<String, String> {
    // 1. Bundled resource (production installs). Tauri v2 layout:
    //    <resource_dir>/resources/bin/terllama  (and terllama.exe on Windows)
    if let Some(rd) = resource_dir {
        let bundled = rd.join("resources").join("bin").join("terllama");
        if bundled.exists() {
            return Ok(bundled.to_string_lossy().to_string());
        }
        let bundled_win = rd.join("resources").join("bin").join("terllama.exe");
        if bundled_win.exists() {
            return Ok(bundled_win.to_string_lossy().to_string());
        }
    }

    // 2. Check PATH first
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            let candidate = format!("{}/terllama", dir);
            if std::path::Path::new(&candidate).exists() {
                return Ok(candidate);
            }
        }
    }

    let candidates = vec![
        "/usr/local/bin/terllama",
        "/usr/bin/terllama",
        "/media/extra/Symlinks/BitNet/terllama-repo/terllama",
    ];
    for c in candidates {
        if std::path::Path::new(c).exists() {
            return Ok(c.to_string());
        }
    }

    Err("terllama binary not found in PATH or common locations".to_string())
}

/// Resolve the HuggingFace repo id for a model registry id.
/// Falls back to the registry id itself if the registry is unreachable.
pub async fn resolve_hf_repo(model_id: &str) -> String {
    match crate::download::fetch_registry().await {
        Ok(registry) => registry
            .models
            .iter()
            .find(|m| m.id == model_id)
            .map(|m| m.hf_repo.clone())
            .unwrap_or_else(|| model_id.to_string()),
        Err(_) => model_id.to_string(),
    }
}
