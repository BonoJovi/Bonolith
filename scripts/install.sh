#!/usr/bin/env bash
# Bonolith installer.
#
# Copies built artifacts to system paths and installs the systemd
# user unit. Build prerequisites first:
#   cargo build --release
#   (cd fcitx5/build && cmake .. -DCMAKE_INSTALL_PREFIX=/usr && make)

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [--no-llm] [--help]

Installs Bonolith. Run from the repository root after building both
the Rust crate and the Fcitx5 addon:

  cargo build --release
  mkdir -p fcitx5/build && cd fcitx5/build
  cmake .. -DCMAKE_INSTALL_PREFIX=/usr && make
  cd ../..
  ./scripts/install.sh

User data at ~/.local/share/bonolith/ is left untouched. On the first
install after the JaIM->Bonolith rebrand, learned data found under the
old ~/.local/share/jaim/ (dict.sqlite + models) is copied across once
(the originals are left in place). If a v1.x user_dict.json or
user_scores.json is present, the Bonolith engine migrates it into
dict.sqlite on first start (renaming the originals to *.migrated).

Options:
  --no-llm  Don't enable bonolith-llm-server.service. Bonolith still works
            (it falls back to the dictionary-only ranker), but no
            local LLM is started. Useful for older PCs. You can
            turn it on later with `bonolith llm on`.
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
check target/release/bonolith                  "run 'cargo build --release'"
check target/release/libbonolith.so            "run 'cargo build --release'"
check fcitx5/build/fcitx5-bonolith.so          "build fcitx5 addon: cd fcitx5/build && cmake .. && make"
check data/bonolith.xml                        "missing source file"
check fcitx5/bonolith-addon.conf               "missing source file"
check fcitx5/bonolith-im.conf                  "missing source file"
check scripts/bonolith-llm-server.service      "missing source file"
if [ "$missing" -eq 1 ]; then
    echo "" >&2
    echo "Install aborted — see missing artifacts above." >&2
    exit 1
fi

echo "Bonolith installer"
echo "=============="

# 1. Stop any currently running Bonolith bits so we can replace files
# cleanly. TERM first to give fcitx5/ibus a chance to save state
# (Fcitx5 will SIGSEGV in AddonManager::saveAll if killed mid-init);
# escalate to KILL only if they linger. Match by exact basename so
# pkill doesn't grep its own argv and kill itself.
#
# ibus-engine-bonolith is killed via -f against its installed path
# rather than -x against the basename: the kernel truncates
# /proc/<pid>/comm to TASK_COMM_LEN (15 chars), so pkill -x sees
# "ibus-engine-jai" (no trailing m) and never matches the 16-char
# basename. Matching the full cmdline path side-steps that and is
# specific enough not to false-positive. This catches orphaned
# engines from before the install (ibus-daemon spawns them as
# children, but they can outlive a daemon TERM and keep running
# the previous binary).
echo "[1/4] Stopping services..."
systemctl --user stop bonolith-llm-server.service >/dev/null 2>&1 || true
sudo pkill -TERM -x ibus-daemon >/dev/null 2>&1 || true
pkill -TERM -f /usr/bin/ibus-engine-bonolith >/dev/null 2>&1 || true
# Also stop the legacy pre-rebrand engine if it's still running, so its
# SQLite handle on ~/.local/share/jaim/ is released before the one-time
# data migration below snapshots it.
pkill -TERM -f /usr/bin/ibus-engine-jaim >/dev/null 2>&1 || true
pkill -TERM -x fcitx5 >/dev/null 2>&1 || true
sleep 2
sudo pkill -KILL -x ibus-daemon >/dev/null 2>&1 || true
pkill -KILL -f /usr/bin/ibus-engine-bonolith >/dev/null 2>&1 || true
pkill -KILL -f /usr/bin/ibus-engine-jaim >/dev/null 2>&1 || true
pkill -KILL -x fcitx5 >/dev/null 2>&1 || true

# 1b. One-time migration from the old JaIM data directory. Pre-rebrand
# installs kept the learned dictionary, user scores and GGUF models under
# ~/.local/share/jaim/; Bonolith reads ~/.local/share/bonolith/. Carry the
# data across once so existing users don't lose their learning. Guarded by
# a marker file so reinstalls never clobber freshly learned Bonolith data
# with the stale JaIM snapshot.
OLD_DATA="$HOME/.local/share/jaim"
NEW_DATA="$HOME/.local/share/bonolith"
if [ -f "$OLD_DATA/dict.sqlite" ] && [ ! -f "$NEW_DATA/.migrated-from-jaim" ]; then
    echo "[1b] Migrating learned data from ~/.local/share/jaim/ ..."
    mkdir -p "$NEW_DATA"
    # Drop any placeholder/empty db so the snapshot lands cleanly.
    rm -f "$NEW_DATA/dict.sqlite" "$NEW_DATA/dict.sqlite-wal" "$NEW_DATA/dict.sqlite-shm"
    if command -v sqlite3 >/dev/null 2>&1; then
        # VACUUM INTO yields a consistent standalone copy that folds in any
        # WAL contents — safe even if a stale -wal/-shm is present.
        sqlite3 "$OLD_DATA/dict.sqlite" "VACUUM INTO '$NEW_DATA/dict.sqlite'"
    else
        # Fallback: copy the db with its WAL/SHM so SQLite recovers on open.
        cp -f "$OLD_DATA/dict.sqlite" "$NEW_DATA/dict.sqlite"
        [ -f "$OLD_DATA/dict.sqlite-wal" ] && cp -f "$OLD_DATA/dict.sqlite-wal" "$NEW_DATA/dict.sqlite-wal"
        [ -f "$OLD_DATA/dict.sqlite-shm" ] && cp -f "$OLD_DATA/dict.sqlite-shm" "$NEW_DATA/dict.sqlite-shm"
    fi
    # Reuse the (large) downloaded GGUF models rather than re-fetching.
    if [ -d "$OLD_DATA/models" ] && [ ! -d "$NEW_DATA/models" ]; then
        cp -r "$OLD_DATA/models" "$NEW_DATA/models"
    fi
    touch "$NEW_DATA/.migrated-from-jaim"
    echo "      done (original ~/.local/share/jaim/ left untouched)."
fi

# 2. System paths (sudo). Use `install -D` so missing parent dirs
# (e.g., /usr/share/fcitx5/inputmethod when Fcitx5 isn't yet
# bootstrapped) are created automatically.
echo "[2/4] Installing system files (sudo required)..."
sudo install -D -m 755 target/release/bonolith         /usr/bin/ibus-engine-bonolith
# `bonolith` is the user-facing CLI name (bonolith llm on/off/status, bonolith
# export/import). IBus invokes the same binary as ibus-engine-bonolith.
sudo ln -sf ibus-engine-bonolith                       /usr/bin/bonolith
sudo install -D -m 644 data/bonolith.xml               /usr/share/ibus/component/bonolith.xml
sudo install -D -m 755 target/release/libbonolith.so   /usr/lib/x86_64-linux-gnu/libbonolith.so
sudo install -D -m 755 fcitx5/build/fcitx5-bonolith.so /usr/lib/x86_64-linux-gnu/fcitx5/fcitx5-bonolith.so
sudo install -D -m 644 fcitx5/bonolith-addon.conf      /usr/share/fcitx5/addon/bonolith.conf
sudo install -D -m 644 fcitx5/bonolith-im.conf         /usr/share/fcitx5/inputmethod/bonolith.conf
sudo install -D -m 755 scripts/bonolith_word_register.py /usr/share/bonolith/scripts/bonolith_word_register.py

# 3. User-level systemd unit. The unit's ExecStartPre handles the
# llama.cpp ggml-backend symlink dance.
echo "[3/4] Installing user systemd unit..."
mkdir -p "$HOME/.config/systemd/user"
install -m 644 scripts/bonolith-llm-server.service "$HOME/.config/systemd/user/bonolith-llm-server.service"

# 4. Bring services back up. Start fcitx5 only if it was registered
# as the user's IM in the past (i.e., a profile exists) — otherwise
# the user is IBus-only and starting fcitx5 would be confusing.
echo "[4/4] Starting services..."
systemctl --user daemon-reload >/dev/null 2>&1 || true
if [ "$NO_LLM" -eq 1 ]; then
    # Make sure any pre-existing enable/start is undone, otherwise
    # --no-llm wouldn't actually keep the LLM off after a reinstall.
    systemctl --user stop bonolith-llm-server.service >/dev/null 2>&1 || true
    systemctl --user disable bonolith-llm-server.service >/dev/null 2>&1 || true
    echo "  (LLM disabled per --no-llm; turn it on later with \`bonolith llm on\`)"
else
    # Ensure the llama-server binary exists before enabling the unit. install.sh
    # only ships the systemd unit; without the binary the engine silently falls
    # back to the heuristic scorer (no real LLM). Fetch it on demand. Skipped
    # entirely under --no-llm (this branch only runs when the LLM is enabled).
    if [ ! -x "$HOME/.local/bin/llama-server" ]; then
        echo "  llama-server not found — fetching prebuilt release..."
        if ! "$SCRIPT_DIR/install-llama-server.sh"; then
            echo "  Warning: llama-server install failed (offline?). The LLM service"
            echo "  won't start until you run scripts/install-llama-server.sh."
        fi
    fi
    systemctl --user enable --now bonolith-llm-server.service >/dev/null 2>&1 || true
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
echo "  bonolith llm status                              # LLM service active/enabled"
echo "  bonolith export /tmp/bonolith-test.json              # should report user entries"

# Warn if `bonolith` on PATH resolves to something other than the symlink
# we just installed. A stale ~/.local/bin/bonolith (or /usr/local/bin/bonolith)
# from a previous build silently shadows /usr/bin/bonolith and produces
# confusing "unknown command 'llm'" errors after upgrades.
if resolved="$(command -v bonolith 2>/dev/null)"; then
    if [ "$resolved" != "/usr/bin/bonolith" ]; then
        echo ""
        echo "Warning: 'bonolith' on your PATH resolves to:"
        echo "    $resolved"
        echo "  not /usr/bin/bonolith. The shadowing copy is likely an older"
        echo "  build and will not have the latest subcommands. Remove it"
        echo "  (e.g. \`rm $resolved\`) so \`bonolith llm on\` runs the freshly"
        echo "  installed binary."
    fi
fi
