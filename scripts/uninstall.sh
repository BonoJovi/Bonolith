#!/usr/bin/env bash
# JaIM uninstaller.
#
# Removes installed binaries, Fcitx5/IBus configs, and the systemd
# user unit. User data at ~/.local/share/jaim/ is preserved by
# default; pass --remove-data to also delete it.

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [--remove-data] [--help]

Removes JaIM from the system. By default user data at
~/.local/share/jaim/ (dictionary, scores, models) is preserved so a
later reinstall picks up the same words and learning history.

Options:
  --remove-data   Also delete ~/.local/share/jaim/ (dict + scores + models)
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

echo "JaIM uninstaller"
echo "================"

# 1. Stop services (best-effort). TERM first so fcitx5 has a chance
# to flush its addon cache cleanly; escalate to KILL only if needed.
echo "[1/5] Stopping services..."
systemctl --user stop jaim-llm-server.service >/dev/null 2>&1 || true
systemctl --user disable jaim-llm-server.service >/dev/null 2>&1 || true
sudo pkill -TERM -f ibus-daemon >/dev/null 2>&1 || true
pkill -TERM -f fcitx5 >/dev/null 2>&1 || true
sleep 2
sudo pkill -KILL -f ibus-daemon >/dev/null 2>&1 || true
pkill -KILL -f fcitx5 >/dev/null 2>&1 || true

# 2. Remove system files
echo "[2/5] Removing system files (sudo required)..."
sudo rm -f \
    /usr/bin/ibus-engine-jaim \
    /usr/share/ibus/component/jaim.xml \
    /usr/lib/x86_64-linux-gnu/libjaim.so \
    /usr/lib/x86_64-linux-gnu/fcitx5/fcitx5-jaim.so \
    /usr/share/fcitx5/addon/jaim.conf \
    /usr/share/fcitx5/inputmethod/jaim.conf

# 3. Remove user-level systemd unit
echo "[3/5] Removing user systemd unit..."
rm -f ~/.config/systemd/user/jaim-llm-server.service
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
    if [ -d "$HOME/.local/share/jaim" ]; then
        rm -rf "$HOME/.local/share/jaim"
        echo "  removed $HOME/.local/share/jaim/"
    fi
else
    if [ -d "$HOME/.local/share/jaim" ]; then
        echo "  preserved $HOME/.local/share/jaim/"
        echo "  (run with --remove-data to also delete dict, scores, models)"
    fi
fi

# Restart IBus so the framework stops looking for the missing engine
echo ""
echo "Restarting IBus..."
ibus-daemon -drx >/dev/null 2>&1 &
disown 2>/dev/null || true

echo ""
echo "Uninstall complete."
