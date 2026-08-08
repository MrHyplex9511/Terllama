#!/usr/bin/env bash
set -euo pipefail

# Terllama — Uninstaller for the full runtime bundle
# Removes the CLI binaries, tokenizer .so, web UI and (optionally) data.

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}━━━ Terllama Uninstall ━━━${NC}"

# ─── Installed bin dir (matches install.sh defaults) ───────────────────
HOME_BIN="${INSTALL_DIR:-$HOME/.local/bin}"
PREFIX="${PREFIX:-$(dirname "$HOME_BIN")}"

for f in terllama terllama-bench terllama-worker terllama-cluster terllama-node \
         libgigatoken_rs.so; do
    if [ -e "$HOME_BIN/$f" ]; then
        rm -f "$HOME_BIN/$f"
        echo -e "  ${GREEN}✓${NC} Removed $HOME_BIN/$f"
    fi
done

if [ -d "$PREFIX/share/terllama" ]; then
    rm -rf "$PREFIX/share/terllama"
    echo -e "  ${GREEN}✓${NC} Removed web UI ($PREFIX/share/terllama)"
fi

# ─── Legacy: old single-binary install ───────────────────────────────
if [ -f /usr/local/bin/terllama ]; then
    if [ -w /usr/local/bin ]; then rm -f /usr/local/bin/terllama
    else sudo rm -f /usr/local/bin/terllama; fi
    echo -e "  ${GREEN}✓${NC} Removed legacy /usr/local/bin/terllama"
fi

# ─── Data directory ──────────────────────────────────────────────────
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