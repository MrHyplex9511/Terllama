use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    pub model_dir: String,
    pub port: u16,
    pub keep_alive: u64,
    pub auto_start: bool,
    pub theme: String,
    pub gguf_mode: bool,
    pub gpu_layers: u32,
    pub cpu_threads: u32,
}

impl Default for Settings {
    fn default() -> Self {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get() as u32)
            .unwrap_or(4);
        Self {
            model_dir: dirs::home_dir()
                .unwrap_or_default()
                .join(".terllama/models")
                .to_string_lossy()
                .to_string(),
            port: 8375,
            keep_alive: 300,
            auto_start: false,
            theme: "dark".to_string(),
            gguf_mode: false,
            gpu_layers: 0,
            cpu_threads: threads,
        }
    }
}

impl Settings {
    pub fn load() -> Self {
        let path = get_config_path();
        if path.exists() {
            std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default()
        } else {
            Self::default()
        }
    }

    pub fn save(&self) -> Result<(), String> {
        let path = get_config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let data = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;

        // Write atomically via a sibling temp file, then rename. On Unix the
        // file is created with owner-only perms (0o600) so the saved settings
        // never sit world-readable. Windows behavior is unchanged.
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, &data).map_err(|e| e.to_string())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))
                .map_err(|e| e.to_string())?;
        }
        std::fs::rename(&tmp, &path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            e.to_string()
        })
    }
}

fn get_config_path() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_default()
        .join(".terllama/desktop-config.json")
}
