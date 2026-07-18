use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Registry {
    pub models: Vec<RegistryModel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryModel {
    pub id: String,
    pub name: String,
    pub hf_repo: String,
    pub format: String,
    pub description: String,
    pub context: u32,
    pub size_mb: u64,
    pub quants: QuantOptions,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantOptions {
    pub ternary: QuantInfo,
    pub q4_k_m: QuantInfo,
    pub q8_0: QuantInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantInfo {
    pub available: bool,
    pub size_mb: u64,
    pub filename: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadedModel {
    pub id: String,
    pub path: String,
    pub size_mb: u64,
    pub quant: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DownloadProgressEvent {
    pub model_id: String,
    pub file: String,
    pub downloaded: u64,
    pub total: u64,
    pub speed: f64,
}

pub async fn fetch_registry() -> Result<Registry, String> {
    let resp = reqwest::get(
        "https://raw.githubusercontent.com/MrHyplex9511/Terllama/main/registry.json",
    )
    .await
    .map_err(|e| format!("Failed to fetch registry: {}", e))?;

    let text = resp
        .text()
        .await
        .map_err(|e| format!("Failed to read registry: {}", e))?;

    serde_json::from_str(&text).map_err(|e| format!("Failed to parse registry: {}", e))
}

pub fn get_models_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_default()
        .join(".terllama/models")
}

pub fn list_downloaded_models() -> Result<Vec<DownloadedModel>, String> {
    let models_dir = get_models_dir();
    if !models_dir.exists() {
        return Ok(vec![]);
    }

    let mut models = vec![];
    for entry in std::fs::read_dir(&models_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            let id = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();
            let total_size: u64 = std::fs::read_dir(&path)
                .map(|entries| {
                    entries
                        .filter_map(|e| e.ok())
                        .filter_map(|e| e.metadata().ok())
                        .map(|m| m.len())
                        .sum()
                })
                .unwrap_or(0);
            models.push(DownloadedModel {
                id,
                path: path.to_string_lossy().to_string(),
                size_mb: total_size / (1024 * 1024),
                quant: "ternary".to_string(),
            });
        }
    }
    Ok(models)
}
