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
Usage: $(basename "$0") [--no-llm] [--help]

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

Options:
  --no-llm  Don't enable jaim-llm-server.service. JaIM still works
            (it falls back to the dictionary-only ranker), but no
            local LLM is started. Useful for older PCs. You can
            turn it on later with `jaim llm on`.
  --help    Show this help.
EOF
}

NO_LLM=0
for arg in "$@"; do
    case "$arg" in
        --no-llm) NO_LLM=1 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; usage >&2; exit 1 ;;
    esac
done

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
# cleanly. TERM first to give fcitx5/ibus a chance to save state
# (Fcitx5 will SIGSEGV in AddonManager::saveAll if killed mid-init);
# escalate to KILL only if they linger. Match by exact basename so
# pkill doesn't grep its own argv and kill itself.
echo "[1/4] Stopping services..."
systemctl --user stop jaim-llm-server.service >/dev/null 2>&1 || true
sudo pkill -TERM -x ibus-daemon >/dev/null 2>&1 || true
pkill -TERM -x fcitx5 >/dev/null 2>&1 || true
sleep 2
sudo pkill -KILL -x ibus-daemon >/dev/null 2>&1 || true
pkill -KILL -x fcitx5 >/dev/null 2>&1 || true

# 2. System paths (sudo). Use `install -D` so missing parent dirs
# (e.g., /usr/share/fcitx5/inputmethod when Fcitx5 isn't yet
# bootstrapped) are created automatically.
echo "[2/4] Installing system files (sudo required)..."
sudo install -D -m 755 target/release/jaim         /usr/bin/ibus-engine-jaim
# `jaim` is the user-facing CLI name (jaim llm on/off/status, jaim
# export/import). IBus invokes the same binary as ibus-engine-jaim.
sudo ln -sf ibus-engine-jaim                       /usr/bin/jaim
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
systemctl --user daemon-reload >/dev/null 2>&1 || true
if [ "$NO_LLM" -eq 1 ]; then
    # Make sure any pre-existing enable/start is undone, otherwise
    # --no-llm wouldn't actually keep the LLM off after a reinstall.
    systemctl --user stop jaim-llm-server.service >/dev/null 2>&1 || true
    systemctl --user disable jaim-llm-server.service >/dev/null 2>&1 || true
    echo "  (LLM disabled per --no-llm; turn it on later with \`jaim llm on\`)"
else
    systemctl --user enable --now jaim-llm-server.service >/dev/null 2>&1 || true
fi

# Detach from the script's controlling TTY so the daemons don't
# inherit the terminal. setsid -f starts a new session with no
# controlling TTY; redirecting all three fds keeps the daemon from
# touching the parent terminal's termios state.
setsid -f ibus-daemon -drx </dev/null >/dev/null 2>&1 || true

if [ -f "$HOME/.config/fcitx5/profile" ]; then
    setsid -f fcitx5 -d </dev/null >/dev/null 2>&1 || true
fi

# Daemons we just spawned (or the systemctl/dbus calls above) can
# leave the controlling tty in -onlcr / no-echo state. Reset it so
# the user's shell prompt comes back cleanly. Best-effort: silently
# skip when stdin isn't a tty (e.g. piped install).
[ -t 0 ] && stty sane 2>/dev/null || true

echo "Install complete."
echo ""
echo "Verify:"
echo "  jaim llm status                              # LLM service active/enabled"
echo "  jaim export /tmp/jaim-test.json              # should report user entries"

# Warn if `jaim` on PATH resolves to something other than the symlink
# we just installed. A stale ~/.local/bin/jaim (or /usr/local/bin/jaim)
# from a previous build silently shadows /usr/bin/jaim and produces
# confusing "unknown command 'llm'" errors after upgrades.
if resolved="$(command -v jaim 2>/dev/null)"; then
    if [ "$resolved" != "/usr/bin/jaim" ]; then
        echo ""
        echo "Warning: 'jaim' on your PATH resolves to:"
        echo "    $resolved"
        echo "  not /usr/bin/jaim. The shadowing copy is likely an older"
        echo "  build and will not have the latest subcommands. Remove it"
        echo "  (e.g. \`rm $resolved\`) so \`jaim llm on\` runs the freshly"
        echo "  installed binary."
    fi
fi
