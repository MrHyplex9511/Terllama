#!/usr/bin/env bash
set -euo pipefail

# Terllama — One-line installer for the full runtime bundle
# =========================================================
# Installs the complete Terllama engine from a release bundle so the
# `terllama` command works from the terminal — including the distributed
# binaries (terllama-node, terllama-worker, terllama-cluster), the
# tokenizer runtime (libgigatoken_rs.so) and the web UI.
#
# Usage:
#   curl -fsSL https://github.com/MrHyplex9511/Terllama/releases/latest/download/install.sh | bash
#   bash install.sh                    # from a downloaded bundle dir
#   INSTALL_DIR=~/.local/bin ./install.sh
#
# Env:
#   INSTALL_DIR   where bin/ contents land (default: ~/.local/bin)
#   PREFIX        where share/ lands (default: derived from INSTALL_DIR)

REPO="MrHyplex9511/Terllama"
VERSION="${1:-latest}"
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"; case "$ARCH" in x86_64|amd64) ARCH="amd64";; aarch64|arm64) ARCH="arm64";; esac

HOME_BIN="${INSTALL_DIR:-$HOME/.local/bin}"
PREFIX="${PREFIX:-$(dirname "$HOME_BIN")}"   # e.g. ~/.local
SHARE_DIR="$PREFIX/share/terllama/web"
mkdir -p "$HOME_BIN" "$SHARE_DIR"

# ─── Where are we? Bundle dir vs single tarball download ─────────────────
BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -x "$BUNDLE_DIR/bin/terllama" ]; then
    echo "📦 Installing from local bundle at $BUNDLE_DIR"
    cp "$BUNDLE_DIR"/bin/terllama* "$HOME_BIN/"
    cp "$BUNDLE_DIR"/bin/libgigatoken_rs.so "$HOME_BIN/" 2>/dev/null || true
    cp -r "$BUNDLE_DIR"/share/terllama/web/* "$SHARE_DIR/" 2>/dev/null || true
else
    # ── Download a release tarball ──────────────────────────────────────
    if [ "$VERSION" = "latest" ]; then
        # Resolve the latest tag via the GitHub API so the asset name
        # matches real tarballs (terllama-<VER>-<OS>-<ARCH>.tar.gz).
        LATEST="$(command -v curl >/dev/null && curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" | grep -oP '"tag_name":\s*"\K[^"]+' || true)"
        [ -n "$LATEST" ] && VERSION="$LATEST"
        URL="https://github.com/$REPO/releases/latest/download/terllama-$VERSION-$OS-$ARCH.tar.gz"
    else
        URL="https://github.com/$REPO/releases/download/$VERSION/terllama-$VERSION-$OS-$ARCH.tar.gz"
    fi
    echo "⬇️  Downloading $URL"
    TMP=$(mktemp -d)
    if command -v curl &>/dev/null; then curl -fsSL "$URL" -o "$TMP/bundle.tar.gz"
    elif command -v wget &>/dev/null; then wget -q "$URL" -O "$TMP/bundle.tar.gz"
    else echo "Need curl or wget"; exit 1; fi
    tar -xzf "$TMP/bundle.tar.gz" -C "$TMP"
    cp "$TMP"/terllama-*/bin/* "$HOME_BIN/"
    cp -r "$TMP"/terllama-*/share/terllama/web/* "$SHARE_DIR/" 2>/dev/null || true
    rm -rf "$TMP"
fi

chmod +x "$HOME_BIN"/terllama "$HOME_BIN"/terllama-node \
         "$HOME_BIN"/terllama-worker "$HOME_BIN"/terllama-cluster \
         "$HOME_BIN"/terllama-bench 2>/dev/null || true

# ── PATH notice ────────────────────────────────────────────────────────
if ! echo ":$PATH:" | grep -q ":$HOME_BIN:"; then
    echo "⚠️  $HOME_BIN is not on your PATH."
    echo "   Add this to ~/.bashrc (or ~/.zshrc):"
    echo "     export PATH=\"$HOME_BIN:\$PATH\""
fi

echo "✅ Terllama installed:"
"$HOME_BIN/terllama" --help | head -3
echo
echo "   Try:  terllama, terllama serve, terllama bench, terllama-node"