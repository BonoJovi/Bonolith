#!/usr/bin/env bash
# Bonolith llama-server bootstrapper.
#
# Fetches a prebuilt llama.cpp CPU release and installs llama-server plus its
# shared libraries so bonolith-llm-server.service can run. The main install.sh
# only installs the systemd unit and assumes the binary already exists; this
# script provides (and reproducibly restores) that binary.
#
# Layout (matches bonolith-llm-server.service):
#   ~/.local/bin/llama-server          the server binary
#   ~/.local/bin/lib*.so*              all shared libs (RUNPATH is $ORIGIN, so
#                                      everything must sit next to the binary)
#   ~/.local/lib/libggml-cpu-*.so      backend libs, also mirrored here so the
#   ~/.local/lib/libggml-rpc.so        unit's ExecStartPre symlink dance works
#
# The CPU build matches the service's `-ngl 0` (no GPU offload).

set -euo pipefail

# Pinned to a known-good release for reproducibility. Override with --tag.
DEFAULT_TAG="b9736"
TAG="$DEFAULT_TAG"
FORCE=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [--tag bNNNN] [--force] [--help]

Downloads the prebuilt llama.cpp CPU release (ubuntu-x64) and installs
llama-server into ~/.local/bin so bonolith-llm-server.service can run.

Options:
  --tag bNNNN  llama.cpp release tag to install (default: $DEFAULT_TAG).
               Use 'latest' to resolve the newest release.
  --force      Reinstall even if the target version is already present.
  --help       Show this help.

After installing, (re)start the service with:
  systemctl --user restart bonolith-llm-server.service
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --tag) TAG="${2:?--tag needs a value}"; shift ;;
        --force) FORCE=1 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
    shift
done

ARCH="$(uname -m)"
if [ "$ARCH" != "x86_64" ]; then
    echo "Unsupported arch: $ARCH (this script installs the ubuntu-x64 build)" >&2
    exit 1
fi

BIN_DIR="$HOME/.local/bin"
LIB_DIR="$HOME/.local/lib"
SERVER_BIN="$BIN_DIR/llama-server"

# Resolve 'latest' to a concrete tag so the rest of the script is uniform.
if [ "$TAG" = "latest" ]; then
    echo "Resolving latest llama.cpp release..."
    TAG="$(curl -fsSL -m 15 https://api.github.com/repos/ggml-org/llama.cpp/releases/latest \
        | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -1)"
    [ -n "$TAG" ] || { echo "Failed to resolve latest tag" >&2; exit 1; }
    echo "  latest = $TAG"
fi

# Skip if the requested version is already installed (unless --force).
if [ "$FORCE" -eq 0 ] && [ -x "$SERVER_BIN" ]; then
    current="$("$SERVER_BIN" --version 2>&1 | sed -n 's/^version: *\([0-9]*\).*/\1/p' | head -1)"
    want="${TAG#b}"
    if [ -n "$current" ] && [ "$current" = "$want" ]; then
        echo "llama-server $TAG already installed at $SERVER_BIN (use --force to reinstall)."
        exit 0
    fi
fi

ASSET="llama-${TAG}-bin-ubuntu-x64.tar.gz"
URL="https://github.com/ggml-org/llama.cpp/releases/download/${TAG}/${ASSET}"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Downloading $ASSET ..."
curl -fSL -m 300 -o "$TMP/ll.tar.gz" "$URL"

echo "Extracting ..."
tar xzf "$TMP/ll.tar.gz" -C "$TMP"
SRC="$(find "$TMP" -maxdepth 2 -name llama-server -type f -printf '%h\n' | head -1)"
[ -n "$SRC" ] || { echo "llama-server not found in archive" >&2; exit 1; }

mkdir -p "$BIN_DIR" "$LIB_DIR"

# RUNPATH is $ORIGIN, so every .so must live next to the binary in BIN_DIR.
echo "Installing to $BIN_DIR ..."
install -m 755 "$SRC/llama-server" "$SERVER_BIN"
for so in "$SRC"/*.so*; do
    install -m 644 "$so" "$BIN_DIR/$(basename "$so")"
done

# Mirror the dlopen'd backend libs into LIB_DIR as well, so the service unit's
# ExecStartPre symlink step has a valid source (and a manual `bonolith llm`
# flow that expects them under ~/.local/lib keeps working).
for so in "$SRC"/libggml-cpu-*.so "$SRC"/libggml-rpc.so; do
    [ -e "$so" ] && install -m 644 "$so" "$LIB_DIR/$(basename "$so")"
done

echo "Verifying ..."
"$SERVER_BIN" --version 2>&1 | head -2

echo ""
echo "Installed llama-server $TAG."
echo "Restart the service:  systemctl --user restart bonolith-llm-server.service"
