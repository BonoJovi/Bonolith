#!/usr/bin/env bash
# Bonolith uninstaller.
#
# Removes installed binaries, Fcitx5/IBus configs, and the systemd
# user unit. User data at ~/.local/share/bonolith/ is preserved by
# default; pass --remove-data to also delete it.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [--remove-data] [--help]

Removes Bonolith from the system. By default user data at
~/.local/share/bonolith/ (dictionary, scores, models) is preserved so a
later reinstall picks up the same words and learning history.

Options:
  --remove-data   Also delete ~/.local/share/bonolith/ (dict + scores + models)
  --help          Show this help
EOF
}

REMOVE_DATA=0
for arg in "$@"; do
    case "$arg" in
        --remove-data) REMOVE_DATA=1 ;;
        --help|-h) usage; exit 0 ;;
        *) echo "Unknown option: $arg" >&2; usage >&2; exit 1 ;;
    esac
done

echo "Bonolith uninstaller"
echo "================"

# 1. Stop services (best-effort). TERM first so fcitx5 has a chance
# to flush its addon cache cleanly; escalate to KILL only if needed.
# Match by exact basename so pkill doesn't kill itself via -f.
echo "[1/5] Stopping services..."
systemctl --user stop bonolith-llm-server.service >/dev/null 2>&1 || true
systemctl --user disable bonolith-llm-server.service >/dev/null 2>&1 || true
sudo pkill -TERM -x ibus-daemon >/dev/null 2>&1 || true
pkill -TERM -x fcitx5 >/dev/null 2>&1 || true
sleep 2
sudo pkill -KILL -x ibus-daemon >/dev/null 2>&1 || true
pkill -KILL -x fcitx5 >/dev/null 2>&1 || true

# 2. Remove system files
echo "[2/5] Removing system files (sudo required)..."
sudo rm -f \
    /usr/bin/ibus-engine-bonolith \
    /usr/share/ibus/component/bonolith.xml \
    /usr/lib/x86_64-linux-gnu/libbonolith.so \
    /usr/lib/x86_64-linux-gnu/fcitx5/fcitx5-bonolith.so \
    /usr/share/fcitx5/addon/bonolith.conf \
    /usr/share/fcitx5/inputmethod/bonolith.conf \
    /usr/share/bonolith/scripts/bonolith_word_register.py
sudo rmdir /usr/share/bonolith/scripts /usr/share/bonolith 2>/dev/null || true

# 3. Remove user-level systemd unit
echo "[3/5] Removing user systemd unit..."
rm -f ~/.config/systemd/user/bonolith-llm-server.service
systemctl --user daemon-reload >/dev/null 2>&1 || true

# 4. Remove ggml backend symlinks created by ExecStartPre.
# Only touch entries that are symlinks into ~/.local/lib/ — leaves
# user-installed real binaries and unrelated files alone.
echo "[4/5] Removing ggml backend symlinks..."
removed=0
for f in "$HOME/.local/bin/libggml-cpu-"*.so "$HOME/.local/bin/libggml-rpc.so"; do
    [ -L "$f" ] || continue
    target="$(readlink -f "$f" 2>/dev/null || true)"
    case "$target" in
        "$HOME/.local/lib/"*)
            rm -f "$f"
            removed=$((removed + 1))
            ;;
    esac
done
echo "  removed $removed symlinks"

# 5. User data
echo "[5/5] User data..."
if [ "$REMOVE_DATA" -eq 1 ]; then
    if [ -d "$HOME/.local/share/bonolith" ]; then
        rm -rf "$HOME/.local/share/bonolith"
        echo "  removed $HOME/.local/share/bonolith/"
    fi
else
    if [ -d "$HOME/.local/share/bonolith" ]; then
        echo "  preserved $HOME/.local/share/bonolith/"
        echo "  (run with --remove-data to also delete dict, scores, models)"
    fi
fi

# Restart IBus so the framework stops looking for the missing engine.
# Detach via setsid so the daemon doesn't inherit the script's TTY.
echo ""
echo "Restarting IBus..."
setsid -f ibus-daemon -drx </dev/null >/dev/null 2>&1 || true

# Reset terminal modes the spawned daemons may have left behind.
[ -t 0 ] && stty sane 2>/dev/null || true

echo ""
echo "Uninstall complete."
