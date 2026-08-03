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
    pub formats: Formats,
}

/// Three download formats: fp (original weights), q4 (GGUF Q4_K_M), ternary (I2_S/ALS).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Formats {
    pub fp: FormatInfo,
    pub q4: FormatInfo,
    pub ternary: FormatInfo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormatInfo {
    pub available: bool,
    pub size_mb: u64,
    #[serde(default)]
    pub filename: String,
    /// Multiple files (for multi-shard FP safetensors downloads).
    #[serde(default)]
    pub files: Vec<String>,
    /// HF repo to download from (defaults to the model's hf_repo when empty).
    #[serde(default)]
    pub hf_repo: String,
    /// Ternary requires local conversion from FP weights.
    #[serde(default)]
    pub needs_conversion: bool,
    #[serde(default)]
    pub note: String,
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

/// File name markers that indicate a directory actually contains model weights.
fn contains_model_files(path: &std::path::Path) -> bool {
    let Ok(entries) = std::fs::read_dir(path) else {
        return false;
    };
    let mut has_any = false;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let lower = name.to_lowercase();
        if lower.ends_with(".gguf")
            || lower.ends_with(".safetensors")
            || lower.ends_with(".bin")
            || lower.ends_with(".i2s")
            || lower.ends_with(".als")
            || lower.contains("model_extra")
            || lower.contains("model_decomposed")
            || lower.contains("model_tq1")
        {
            has_any = true;
        }
        // Don't count partial downloads (.part) or tokenizer-only dirs.
        if lower.ends_with(".part") || lower == "tokenizer.json" || lower == "config.json" {
            continue;
        }
    }
    has_any
}

/// Detect the download format from the files present in a model dir.
/// Returns "fp", "q4", or "ternary".
fn detect_quant(path: &std::path::Path) -> String {
    let Ok(entries) = std::fs::read_dir(path) else {
        return "ternary".to_string();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();

    // Sort so .part / tokenizer files don't win detection.
    names.sort();

    let mut saw_safetensors = false;
    let mut saw_gguf = false;
    let mut saw_q4 = false;
    let mut saw_q2 = false;
    let mut saw_fp_gguf = false;

    for name in &names {
        let lower = name.to_lowercase();
        if lower.ends_with(".part") || lower.ends_with(".json") {
            continue;
        }
        if lower.ends_with(".safetensors") {
            saw_safetensors = true;
        }
        if lower.ends_with(".gguf") {
            saw_gguf = true;
            if lower.contains("q4") || lower.contains("q5") || lower.contains("q3") {
                saw_q4 = true;
            } else if lower.contains("q2") {
                saw_q2 = true;
            } else if lower.contains("f16") || lower.contains("f32") || lower.contains("fp16") || lower.contains("fp32") {
                saw_fp_gguf = true;
            }
        }
        // Ternary native markers: model_extra.bin / model_decomposed_*.bin / model_tq1.bin / .i2s / .als
        if lower.contains("model_extra")
            || lower.contains("model_decomposed")
            || lower.contains("model_tq1")
            || lower.ends_with(".i2s")
            || lower.ends_with(".als")
        {
            return "ternary".to_string();
        }
    }

    if saw_q4 {
        return "q4".to_string();
    }
    if saw_fp_gguf {
        return "fp".to_string();
    }
    if saw_safetensors {
        return "fp".to_string();
    }
    if saw_q2 {
        return "ternary".to_string();
    }
    if saw_gguf {
        return "ternary".to_string();
    }
    "ternary".to_string()
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
        if !path.is_dir() {
            continue;
        }
        // STALE-STATE FIX: skip dirs with no actual model files (empty leftover dirs
        // from the old slug-naming bug used to show as "installed" but had no weights).
        if !contains_model_files(&path) {
            continue;
        }

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
                    .filter(|m| m.is_file())
                    .map(|m| m.len())
                    .sum()
            })
            .unwrap_or(0);
        let quant = detect_quant(&path);
        models.push(DownloadedModel {
            id,
            path: path.to_string_lossy().to_string(),
            size_mb: total_size / (1024 * 1024),
            quant,
        });
    }
    Ok(models)
}
