use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tauri::Emitter;
use tauri::Manager;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvertProgress {
    pub line: String,
    pub done: bool,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvertConfig {
    pub model: String,        // HF model name or local path
    pub format: String,       // "i2s" or "als"
    pub terms: u32,           // ALS terms (default 12)
    pub out_dir: String,      // output directory
}

/// Find Python 3 binary on the system
pub fn find_python() -> Option<PathBuf> {
    let candidates = ["python3", "python"];
    for cmd in &candidates {
        let out = std::process::Command::new("which")
            .arg(cmd)
            .output()
            .ok()?;
        if out.status.success() {
            let path = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !path.is_empty() {
                return Some(PathBuf::from(path));
            }
        }
    }

    // Also check common paths directly
    let common = [
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "/usr/bin/python",
        "/usr/local/bin/python",
    ];
    for p in &common {
        if std::path::Path::new(p).exists() {
            return Some(PathBuf::from(p));
        }
    }
    None
}

/// Check Python version meets minimum
pub fn check_python_version(python: &PathBuf) -> Result<(u32, u32), String> {
    let out = std::process::Command::new(python)
        .arg("--version")
        .output()
        .map_err(|e| format!("Failed to run Python: {}", e))?;

    let ver_str = String::from_utf8_lossy(&out.stdout);
    // Parse "Python 3.X.Y"
    let parts: Vec<&str> = ver_str.split_whitespace().collect();
    if parts.len() < 2 {
        return Err(format!("Could not parse Python version: {}", ver_str));
    }
    let nums: Vec<&str> = parts[1].split('.').collect();
    if nums.len() < 2 {
        return Err(format!("Could not parse version number: {}", parts[1]));
    }
    let major: u32 = nums[0].parse().map_err(|_| "Invalid major version")?;
    let minor: u32 = nums[1].parse().map_err(|_| "Invalid minor version")?;
    Ok((major, minor))
}

/// Get the directory where conversion scripts are bundled
pub fn get_scripts_dir(app_handle: &tauri::AppHandle) -> Result<PathBuf, String> {
    // In development, scripts are in src-tauri/resources/scripts/
    // In production, they're bundled in the AppImage/deb resources
    let resource_dir = app_handle
        .path()
        .resource_dir()
        .map_err(|e| format!("Failed to get resource dir: {}", e))?;

    let scripts_dir = resource_dir.join("scripts");
    if scripts_dir.exists() {
        return Ok(scripts_dir);
    }

    // Fallback: try the repo scripts directory (development)
    let dev_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("resources")
        .join("scripts");
    if dev_path.exists() {
        return Ok(dev_path);
    }

    Err("Conversion scripts not found. Reinstall the application.".to_string())
}

/// Run the conversion as a subprocess, streaming output via Tauri events
pub async fn run_conversion(
    app_handle: tauri::AppHandle,
    cancel_flag: Arc<AtomicBool>,
    config: ConvertConfig,
) -> Result<(), String> {
    let python = find_python().ok_or_else(|| {
        "Python 3 not found. Install Python 3.10+ to use model conversion.".to_string()
    })?;

    let (major, minor) = check_python_version(&python)?;
    if major < 3 || (major == 3 && minor < 10) {
        return Err(format!(
            "Python 3.10+ required, found {}.{}",
            major, minor
        ));
    }

    let scripts_dir = get_scripts_dir(&app_handle)?;
    let script_path = scripts_dir.join("export_ternary_model_bitnet.py");

    if !script_path.exists() {
        return Err(format!(
            "Conversion script not found at {}",
            script_path.display()
        ));
    }

    // Ensure output directory exists
    let out_dir = PathBuf::from(&config.out_dir);
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("Failed to create output directory: {}", e))?;

    // Check for required Python packages first
    let check_result = std::process::Command::new(&python)
        .args(["-c", "import torch, transformers; print('ok')"])
        .output()
        .map_err(|e| format!("Failed to check Python packages: {}", e))?;

    if !check_result.status.success() {
        let stderr = String::from_utf8_lossy(&check_result.stderr);
        return Err(format!(
            "Missing required Python packages (torch, transformers).\nInstall: pip install torch transformers\nError: {}",
            stderr.lines().next().unwrap_or("unknown")
        ));
    }

    // Build command
    let mut cmd = tokio::process::Command::new(&python);
    cmd.arg(&script_path)
        .arg("--model")
        .arg(&config.model)
        .arg("--format")
        .arg(&config.format)
        .arg("--outdir")
        .arg(&config.out_dir);

    if config.format == "als" && config.terms > 0 {
        cmd.arg("--terms").arg(config.terms.to_string());
    }

    // Spawn with piped stdout/stderr
    cmd.stdout(std::process::Stdio::piped());
    cmd.stderr(std::process::Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| format!("Failed to start conversion: {}", e))?;

    let stdout = child.stdout.take().ok_or("No stdout")?;
    let stderr = child.stderr.take().ok_or("No stderr")?;

    let app_handle_clone = app_handle.clone();
    let cancel_clone = cancel_flag.clone();

    // Stream stdout
    let stdout_handle = tokio::spawn(async move {
        use tokio::io::AsyncBufReadExt;
        let reader = tokio::io::BufReader::new(stdout);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if cancel_clone.load(Ordering::Relaxed) {
                break;
            }
            let _ = app_handle_clone.emit("convert-progress", ConvertProgress {
                line,
                done: false,
                error: None,
            });
        }
    });

    // Stream stderr
    let app_handle_clone2 = app_handle.clone();
    let cancel_clone2 = cancel_flag.clone();
    let stderr_handle = tokio::spawn(async move {
        use tokio::io::AsyncBufReadExt;
        let reader = tokio::io::BufReader::new(stderr);
        let mut lines = reader.lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if cancel_clone2.load(Ordering::Relaxed) {
                break;
            }
            let _ = app_handle_clone2.emit("convert-progress", ConvertProgress {
                line,
                done: false,
                error: None,
            });
        }
    });

    // Wait for completion or cancellation
    let status = loop {
        tokio::select! {
            status = child.wait() => break status.map_err(|e| format!("Process error: {}", e))?,
            _ = tokio::time::sleep(tokio::time::Duration::from_millis(100)) => {
                if cancel_flag.load(Ordering::Relaxed) {
                    let _ = child.kill().await;
                    // Signal cancellation
                    let _ = app_handle.emit("convert-progress", ConvertProgress {
                        line: "\n⚠️ Conversion cancelled by user.".to_string(),
                        done: true,
                        error: Some("Cancelled".to_string()),
                    });
                    return Ok(());
                }
            }
        }
    };

    // Wait for streamers to finish
    let _ = stdout_handle.await;
    let _ = stderr_handle.await;

    if status.success() {
        let _ = app_handle.emit("convert-progress", ConvertProgress {
            line: "\n✅ Conversion complete! Model is ready in Library.".to_string(),
            done: true,
            error: None,
        });
        Ok(())
    } else {
        let _ = app_handle.emit("convert-progress", ConvertProgress {
            line: "\n❌ Conversion failed.".to_string(),
            done: true,
            error: Some("Process exited with error".to_string()),
        });
        Err("Conversion process failed".to_string())
    }
}
