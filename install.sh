#!/usr/bin/env bash
set -euo pipefail

# Terllama — One-line install
# Fetches latest pre-built binary from GitHub Releases

REPO="MrHyplex9511/Terllama"
VERSION="${1:-latest}"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"

# Detect OS/ARCH
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)  OS="linux" ;;
    Darwin) OS="darwin" ;;
    *)      echo "Unsupported OS: $OS"; exit 1 ;;
esac

case "$ARCH" in
    x86_64|amd64) ARCH="amd64" ;;
    aarch64|arm64) ARCH="arm64" ;;
    *) echo "Unsupported arch: $ARCH"; exit 1 ;;
esac

# Determine download URL
if [ "$VERSION" = "latest" ]; then
    DOWNLOAD_URL="https://github.com/$REPO/releases/latest/download/terllama-$OS-$ARCH"
else
    DOWNLOAD_URL="https://github.com/$REPO/releases/download/$VERSION/terllama-$OS-$ARCH"
fi

echo "📦 Downloading Terllama ($OS-$ARCH)..."
TMPFILE=$(mktemp)
if command -v curl &>/dev/null; then
    curl -fsSL "$DOWNLOAD_URL" -o "$TMPFILE"
elif command -v wget &>/dev/null; then
    wget -q "$DOWNLOAD_URL" -O "$TMPFILE"
else
    echo "Need curl or wget"; exit 1
fi

chmod +x "$TMPFILE"

if [ -w "$INSTALL_DIR" ]; then
    mv "$TMPFILE" "$INSTALL_DIR/terllama"
else
    echo "Elevated permissions needed for $INSTALL_DIR"
    sudo mv "$TMPFILE" "$INSTALL_DIR/terllama"
fi

echo "✅ Terllama installed to $INSTALL_DIR/terllama"
echo "   Run: terllama --help"
