#!/usr/bin/env bash
set -euo pipefail

# Terllama — Uninstaller
# Removes CLI binary, data directory, and optionally the Desktop app

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}━━━ Terllama Uninstall ━━━${NC}"

# ─── CLI binary ──────────────────────────────────────────────
CLI_BIN="/usr/local/bin/terllama"
if [ -f "$CLI_BIN" ]; then
    if [ -w "$(dirname "$CLI_BIN")" ]; then
        rm "$CLI_BIN"
    else
        sudo rm "$CLI_BIN"
    fi
    echo -e "  ${GREEN}✓${NC} Removed CLI binary: $CLI_BIN"
else
    echo -e "  ${YELLOW}−${NC} CLI binary not found at $CLI_BIN"
fi

# ─── Desktop app (APT) ───────────────────────────────────────
if dpkg -s terllama-desktop &>/dev/null 2>&1; then
    echo ""
    echo -e "  Desktop app (deb) detected. Remove? [y/N] "
    read -r resp
    if [[ "$resp" =~ ^[Yy]$ ]]; then
        sudo apt remove -y terllama-desktop
        echo -e "  ${GREEN}✓${NC} Removed desktop app"
    fi
fi

# ─── Desktop app (RPM) ───────────────────────────────────────
if rpm -q terllama-desktop &>/dev/null 2>&1; then
    echo ""
    echo -e "  Desktop app (rpm) detected. Remove? [y/N] "
    read -r resp
    if [[ "$resp" =~ ^[Yy]$ ]]; then
        sudo rpm -e terllama-desktop
        echo -e "  ${GREEN}✓${NC} Removed desktop app"
    fi
fi

# ─── Data directory ──────────────────────────────────────────
DATA_DIR="${HOME}/.terllama"
if [ -d "$DATA_DIR" ]; then
    echo ""
    echo -e "  Remove all models and data in ${DATA_DIR}? [y/N] "
    read -r resp
    if [[ "$resp" =~ ^[Yy]$ ]]; then
        rm -rf "$DATA_DIR"
        echo -e "  ${GREEN}✓${NC} Removed $DATA_DIR"
    fi
fi

echo ""
echo -e "${GREEN}✅ Terllama uninstalled.${NC}"
