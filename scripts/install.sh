#!/usr/bin/env bash
# JaIM installer.
#
# Copies built artifacts to system paths and installs the systemd
# user unit. Build prerequisites first:
#   cargo build --release
#   (cd fcitx5/build && cmake .. -DCMAKE_INSTALL_PREFIX=/usr && make)

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [--help]

Installs JaIM. Run from the repository root after building both
the Rust crate and the Fcitx5 addon:

  cargo build --release
  mkdir -p fcitx5/build && cd fcitx5/build
  cmake .. -DCMAKE_INSTALL_PREFIX=/usr && make
  cd ../..
  ./scripts/install.sh

User data at ~/.local/share/jaim/ is left untouched. If a v1.x
user_dict.json or user_scores.json is found there, the JaIM engine
will migrate it into dict.sqlite on first start (renaming the
originals to *.migrated).
EOF
}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    usage; exit 0
fi

# Resolve repo root from script location so the installer works
# regardless of cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$REPO_ROOT"

# Verify build artifacts up front so we don't fail half-installed.
missing=0
check() {
    if [ ! -f "$1" ]; then
        echo "Missing: $1 ($2)" >&2
        missing=1
    fi
}
check target/release/jaim                  "run 'cargo build --release'"
check target/release/libjaim.so            "run 'cargo build --release'"
check fcitx5/build/fcitx5-jaim.so          "build fcitx5 addon: cd fcitx5/build && cmake .. && make"
check data/jaim.xml                        "missing source file"
check fcitx5/jaim-addon.conf               "missing source file"
check fcitx5/jaim-im.conf                  "missing source file"
check scripts/jaim-llm-server.service      "missing source file"
if [ "$missing" -eq 1 ]; then
    echo "" >&2
    echo "Install aborted — see missing artifacts above." >&2
    exit 1
fi

echo "JaIM installer"
echo "=============="

# 1. Stop any currently running JaIM bits so we can replace files
# cleanly. Use TERM first so fcitx5/ibus get a chance to save state
# (Fcitx5 will SIGSEGV in AddonManager::saveAll if killed mid-init);
# escalate to KILL only if they linger.
echo "[1/4] Stopping services..."
systemctl --user stop jaim-llm-server.service >/dev/null 2>&1 || true
sudo pkill -TERM -f ibus-daemon >/dev/null 2>&1 || true
pkill -TERM -f fcitx5 >/dev/null 2>&1 || true
sleep 2
sudo pkill -KILL -f ibus-daemon >/dev/null 2>&1 || true
pkill -KILL -f fcitx5 >/dev/null 2>&1 || true

# 2. System paths (sudo). Use `install -D` so missing parent dirs
# (e.g., /usr/share/fcitx5/inputmethod when Fcitx5 isn't yet
# bootstrapped) are created automatically.
echo "[2/4] Installing system files (sudo required)..."
sudo install -D -m 755 target/release/jaim         /usr/bin/ibus-engine-jaim
sudo install -D -m 644 data/jaim.xml               /usr/share/ibus/component/jaim.xml
sudo install -D -m 755 target/release/libjaim.so   /usr/lib/x86_64-linux-gnu/libjaim.so
sudo install -D -m 755 fcitx5/build/fcitx5-jaim.so /usr/lib/x86_64-linux-gnu/fcitx5/fcitx5-jaim.so
sudo install -D -m 644 fcitx5/jaim-addon.conf      /usr/share/fcitx5/addon/jaim.conf
sudo install -D -m 644 fcitx5/jaim-im.conf         /usr/share/fcitx5/inputmethod/jaim.conf

# 3. User-level systemd unit. The unit's ExecStartPre handles the
# llama.cpp ggml-backend symlink dance.
echo "[3/4] Installing user systemd unit..."
mkdir -p "$HOME/.config/systemd/user"
install -m 644 scripts/jaim-llm-server.service "$HOME/.config/systemd/user/jaim-llm-server.service"

# 4. Bring services back up. Start fcitx5 only if it was registered
# as the user's IM in the past (i.e., a profile exists) — otherwise
# the user is IBus-only and starting fcitx5 would be confusing.
echo "[4/4] Starting services..."
systemctl --user daemon-reload
systemctl --user enable --now jaim-llm-server.service
ibus-daemon -drx >/dev/null 2>&1 &
disown 2>/dev/null || true

if [ -f "$HOME/.config/fcitx5/profile" ]; then
    fcitx5 -d >/dev/null 2>&1 &
    disown 2>/dev/null || true
fi

echo ""
echo "Install complete."
echo ""
echo "Verify:"
echo "  ibus-engine-jaim export /tmp/jaim-test.json   # should report user entries"
echo "  systemctl --user status jaim-llm-server.service"
echo ""
echo "If Fcitx5's menu shows wrong input methods after install,"
echo "clear its cache and restart:"
echo "  pkill -9 fcitx5; rm -rf ~/.cache/fcitx5; fcitx5 -d &"
