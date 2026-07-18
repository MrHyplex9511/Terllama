# Terllama — Windows Uninstaller
param([switch]$Force)

$ErrorActionPreference = "Stop"

Write-Host "━━━ Terllama Uninstall ━━━" -ForegroundColor Yellow

# ─── CLI binary ──────────────────────────────────────────────
$CliPath = "$env:ProgramFiles\Terllama\terllama.exe"
if (Test-Path $CliPath) {
    Remove-Item -Path "$env:ProgramFiles\Terllama" -Recurse -Force
    Write-Host "  ✓ Removed CLI binary" -ForegroundColor Green
} else {
    Write-Host "  − CLI binary not found" -ForegroundColor Yellow
}

# ─── Desktop app (MSI) ───────────────────────────────────────
$DesktopPath = "$env:ProgramFiles\Terllama Desktop"
if (Test-Path $DesktopPath) {
    if ($Force) {
        $resp = "y"
    } else {
        $resp = Read-Host "Remove desktop app? [y/N]"
    }
    if ($resp -eq "y" -or $resp -eq "Y") {
        # Find MSI product code and uninstall
        $app = Get-WmiObject -Class Win32_Product | Where-Object { $_.Name -like "*Terllama*" }
        if ($app) {
            $app.Uninstall()
            Write-Host "  ✓ Removed desktop app" -ForegroundColor Green
        } else {
            Remove-Item -Path $DesktopPath -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# ─── Data directory ──────────────────────────────────────────
$DataDir = "$env:USERPROFILE\.terllama"
if (Test-Path $DataDir) {
    if ($Force) {
        $resp = "y"
    } else {
        $resp = Read-Host "Remove all models and data? [y/N]"
    }
    if ($resp -eq "y" -or $resp -eq "Y") {
        Remove-Item -Path $DataDir -Recurse -Force
        Write-Host "  ✓ Removed data directory" -ForegroundColor Green
    }
}

# ─── PATH cleanup ────────────────────────────────────────────
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -like "*Terllama*") {
    $NewPath = ($UserPath -split ';' | Where-Object { $_ -notlike "*Terllama*" }) -join ';'
    [Environment]::SetEnvironmentVariable("Path", $NewPath, "User")
    Write-Host "  ✓ Cleaned PATH" -ForegroundColor Green
}

Write-Host "✅ Terllama uninstalled." -ForegroundColor Green
