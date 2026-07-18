use serde::{Deserialize, Serialize};
use std::process::{Child, Command};
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

    pub async fn start(&self, model_id: String, port: u16) -> Result<(), String> {
        {
            let mut status = self.status.lock().map_err(|e| e.to_string())?;
            *status = ServerStatus::Starting;
        }

        let models_dir = dirs::home_dir()
            .unwrap_or_default()
            .join(".terllama/models")
            .join(&model_id);

        let binary = find_terllama_binary()?;

        let mut child = Command::new(&binary)
            .arg("serve")
            .arg("--port")
            .arg(port.to_string())
            .env(
                "TERLLAMA_MODEL_DIR",
                models_dir.to_string_lossy().to_string(),
            )
            .env("TERLLAMA_PORT", port.to_string())
            .spawn()
            .map_err(|e| format!("Failed to start server: {}", e))?;

        // Poll health endpoint
        let health_url = format!("http://127.0.0.1:{}/health", port);
        let max_retries = 60u32;
        let mut started = false;

        for _ in 0..max_retries {
            tokio::time::sleep(Duration::from_millis(500)).await;

            // Check if process is still alive
            if let Ok(Some(_)) = child.try_wait() {
                break;
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
            *status = ServerStatus::Error("Server failed to start within 30s".to_string());
            Err("Server failed to start within 30 seconds".to_string())
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

pub fn find_terllama_binary() -> Result<String, String> {
    // Check PATH first
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
