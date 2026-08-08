#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Terllama release packager
#
# Builds a Release build and assembles a portable runtime bundle:
#
#   terllama-<VERSION>-<OS>-<ARCH>/
#   ├── bin/                        # all 5 executables + libgigatoken_rs.so
#   │   ├── terllama
#   │   ├── terllama-bench
#   │   ├── terllama-worker
#   │   ├── terllama-cluster
#   │   ├── terllama-node
#   │   └── libgigatoken_rs.so
#   ├── share/terllama/web/         # web UI (index.html, network.js, ...)
#   ├── install.sh                  # installs the bundle onto PATH
#   └── uninstall.sh
#
# The manager binary (terllama/terllama-node) resolves all sibling
# executables and the tokenizer .so relative to its own location, so the
# whole runtime must ship together in bin/.
#
# Usage:  scripts/release.sh [version]        (default: TERLLAMA_VERSION)
# Env:    BUILD_DIR=/tmp/terllama-release     (default: build-release)
#
# Produces:  dist/terllama-<VERSION>-<OS>-<ARCH>.tar.gz
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/.."

VERSION="${1:-v$(grep -oP 'TERLLAMA_VERSION "\K[^"]+' src/cli/commands.h)}"
BUILD_DIR="${BUILD_DIR:-build-release}"
DIST_DIR="dist"

OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"; [ "$ARCH" = "x86_64" ] && ARCH="amd64"
BUNDLE="terllama-${VERSION}-${OS}-${ARCH}"
STAGE="$DIST_DIR/$BUNDLE"

echo "═══ Terllama release bundle: ${VERSION} (${OS}-${ARCH}) ═══"

# ─── 1. Build -----------------------------------------------------------------
echo "▸ Building Release..."
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARK=ON >/dev/null
cmake --build "$BUILD_DIR" -j"$(nproc)" >/dev/null

# ─── 2. Cargo .so (tokenizer, Python-free C ABI) ------------------------------
if [ ! -f "$BUILD_DIR/libgigatoken_rs.so" ]; then
    echo "▸ Building GigaToken .so from source..."
    (cd third_party/gigatoken-capi && cargo build --release -p gigatoken-capi)
    cp third_party/gigatoken-capi/target/release/libgigatoken_rs.so "$BUILD_DIR/"
fi

# ─── 3. Assemble staging tree -------------------------------------------------
rm -rf "$STAGE"
mkdir -p "$STAGE/bin" "$STAGE/share/terllama/web"

for t in terllama terllama-bench terllama-worker terllama-cluster terllama-node; do
    [ -f "$BUILD_DIR/$t" ] && cp "$BUILD_DIR/$t" "$STAGE/bin/"
done
cp "$BUILD_DIR/libgigatoken_rs.so" "$STAGE/bin/"
cp -r web/index.html web/network.js web/anime.min.js "$STAGE/share/terllama/web/"

# ─── 4. Ship the install scripts with the bundle ------------------------------
cp install.sh uninstall.sh "$STAGE/"

echo "▸ Bundle contents:"
find "$STAGE" -type f | sort | sed 's/^/    /'

# ─── 5. Package -----------------------------------------------------------------
mkdir -p "$DIST_DIR"
tar -C "$DIST_DIR" -czf "$DIST_DIR/$BUNDLE.tar.gz" "$BUNDLE"
echo "✔ Wrote: $DIST_DIR/$BUNDLE.tar.gz"

# ─── 6. Verify .so resolves ---------------------------------------------
if ldd "$STAGE/bin/terllama" 2>/dev/null | grep -q "not found"; then
    echo "⚠  Missing shared libs in terllama (see ldd above)" >&2
fi