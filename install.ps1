# Terllama — Windows one-line install
param(
    [string]$Version = "latest",
    [string]$InstallDir = "$env:ProgramFiles\Terllama"
)

# Fail loudly on any error instead of printing false success.
$ErrorActionPreference = "Stop"

$Repo = "MrHyplex9511/Terllama"
$Arch = if ([Environment]::Is64BitOperatingSystem) { "amd64" } else { Write-Error "32-bit not supported"; exit 1 }

if ($Version -eq "latest") {
    $Url = "https://github.com/$Repo/releases/latest/download/terllama-windows-$Arch.exe"
} else {
    $Url = "https://github.com/$Repo/releases/download/$Version/terllama-windows-$Arch.exe"
}

Write-Host "📦 Downloading Terllama (windows-$Arch)..." -ForegroundColor Cyan
# Unique temp name: avoids clobbering concurrent installs / leftover files.
$OutPath = Join-Path $env:TEMP ("terllama-" + [guid]::NewGuid().ToString() + ".exe")

try {
    Invoke-WebRequest -Uri $Url -OutFile $OutPath

    New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
    Move-Item -Path $OutPath -Destination "$InstallDir\terllama.exe" -Force

    # Move-Item can fail silently for non-elevated users writing into
    # protected dirs (e.g. Program Files); verify before reporting success.
    if (-not (Test-Path -LiteralPath "$InstallDir\terllama.exe")) {
        Write-Error "Terllama install failed: destination file missing after move. Check write access to $InstallDir (re-run as Administrator)."
        exit 1
    }
} catch {
    Write-Error "Terllama install failed: $($_.Exception.Message)"
    exit 1
}

# Add to PATH if not already
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$InstallDir", "User")
    Write-Host "✅ Added $InstallDir to PATH (re-login required)" -ForegroundColor Green
}

Write-Host "✅ Terllama installed to $InstallDir\terllama.exe" -ForegroundColor Green
Write-Host "   Run: terllama --help"
