# Terllama — Windows one-line install
param(
    [string]$Version = "latest",
    [string]$InstallDir = "$env:ProgramFiles\Terllama"
)

$Repo = "MrHyplex9511/Terllama"
$Arch = if ([Environment]::Is64BitOperatingSystem) { "amd64" } else { Write-Error "32-bit not supported"; exit 1 }

if ($Version -eq "latest") {
    $Url = "https://github.com/$Repo/releases/latest/download/terllama-windows-$Arch.exe"
} else {
    $Url = "https://github.com/$Repo/releases/download/$Version/terllama-windows-$Arch.exe"
}

Write-Host "📦 Downloading Terllama (windows-$Arch)..." -ForegroundColor Cyan
$OutPath = "$env:TEMP\terllama.exe"
Invoke-WebRequest -Uri $Url -OutFile $OutPath

New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
Move-Item -Path $OutPath -Destination "$InstallDir\terllama.exe" -Force

# Add to PATH if not already
$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$InstallDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$InstallDir", "User")
    Write-Host "✅ Added $InstallDir to PATH (re-login required)" -ForegroundColor Green
}

Write-Host "✅ Terllama installed to $InstallDir\terllama.exe" -ForegroundColor Green
Write-Host "   Run: terllama --help"
