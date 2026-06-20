#!/usr/bin/python3
# Use the system python3 explicitly so we pick up python3-gi from the
# distro package (linuxbrew's python3 typically lacks PyGObject bindings).
"""JaIM word register / edit dialog (GTK3).

Used in place of `zenity --forms` so we can re-activate the host IME each
time focus enters an Entry. GTK creates a fresh input context per
GtkEntry, which is exactly why zenity loses 日本語 ON after Tab; here we
hook focus-in-event on every Entry and trigger an IME-activate action.

Args (positional + optional):
    backend             "ibus" or "fcitx5"
    --mode {register,edit}
                        Register (default): empty fields, title 単語登録.
                        Edit: prefill fields, title 単語編集.
    --reading TEXT      Initial reading (edit mode)
    --surface TEXT      Initial surface (edit mode)

stdout: "<reading>|<surface>" on OK, nothing on Cancel.
exit:   0 on OK, 1 on Cancel / invalid args.
"""

import argparse
import subprocess
import sys

import gi

gi.require_version("Gtk", "3.0")
from gi.repository import GLib, Gtk  # noqa: E402


def _activate_ibus():
    """Simulate JaIM's IME toggle hotkey (Ctrl+Shift+space) via xdotool.

    IBus' DBus API (Bus.set_global_engine / InputContext.enable /
    InputContext.set_engine) does not actually flip a focused IC into
    日本語ON state when called from another process — empirically all
    three are no-ops here. Instead, inject the key that the engine
    itself maps to "toggle IME". Each GtkEntry gets a fresh IC that
    starts in OFF, so a single toggle on focus-in always lands in ON.
    """
    try:
        subprocess.Popen(
            ["xdotool", "key", "ctrl+shift+space"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"ibus activate failed: {exc}", file=sys.stderr)


def _activate_fcitx5():
    """Activate focused input context via fcitx5-remote CLI."""
    try:
        subprocess.Popen(
            ["fcitx5-remote", "-o"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception as exc:
        print(f"fcitx5 activate failed: {exc}", file=sys.stderr)


def make_activate(backend):
    if backend == "ibus":
        return _activate_ibus
    if backend == "fcitx5":
        return _activate_fcitx5
    raise ValueError(f"unknown backend: {backend!r}")


class WordDialog(Gtk.Dialog):
    def __init__(self, activate, mode, init_reading, init_surface):
        title = "JaIM: 単語登録" if mode == "register" else "JaIM: 単語編集"
        super().__init__(title=title)
        self.set_default_size(420, 0)
        self.set_resizable(False)
        self.add_buttons(
            "キャンセル", Gtk.ResponseType.CANCEL,
            "OK", Gtk.ResponseType.OK,
        )
        self.set_default_response(Gtk.ResponseType.OK)

        self.activate = activate

        box = self.get_content_area()
        box.set_spacing(8)
        box.set_border_width(12)

        intro_text = (
            "ユーザー辞書に新しい単語を登録します"
            if mode == "register"
            else "ユーザー辞書のエントリを編集します"
        )
        intro = Gtk.Label(label=intro_text, halign=Gtk.Align.START)
        box.pack_start(intro, False, False, 0)

        grid = Gtk.Grid(column_spacing=10, row_spacing=8)
        box.pack_start(grid, True, True, 0)

        grid.attach(Gtk.Label(label="よみ (ひらがな):", halign=Gtk.Align.START), 0, 0, 1, 1)
        self.reading_entry = Gtk.Entry(hexpand=True)
        self.reading_entry.set_activates_default(True)
        if init_reading:
            self.reading_entry.set_text(init_reading)
        grid.attach(self.reading_entry, 1, 0, 1, 1)

        grid.attach(Gtk.Label(label="単語:", halign=Gtk.Align.START), 0, 1, 1, 1)
        self.surface_entry = Gtk.Entry(hexpand=True)
        self.surface_entry.set_activates_default(True)
        if init_surface:
            self.surface_entry.set_text(init_surface)
        grid.attach(self.surface_entry, 1, 1, 1, 1)

        self.reading_entry.connect("focus-in-event", self._on_focus_in)
        self.surface_entry.connect("focus-in-event", self._on_focus_in)

        self.show_all()

    def _on_focus_in(self, _widget, _event):
        # Defer briefly so the GtkEntry's IM context has time to notify the
        # host IME daemon of the focus change before we inject the toggle.
        GLib.timeout_add(80, self._activate_once)
        return False  # propagate

    def _activate_once(self):
        self.activate()
        return False  # one-shot


def main(argv):
    parser = argparse.ArgumentParser(prog="jaim_word_register.py")
    parser.add_argument("backend", choices=["ibus", "fcitx5"])
    parser.add_argument("--mode", choices=["register", "edit"], default="register")
    parser.add_argument("--reading", default="")
    parser.add_argument("--surface", default="")
    args = parser.parse_args(argv[1:])

    activate = make_activate(args.backend)

    dialog = WordDialog(activate, args.mode, args.reading, args.surface)
    response = dialog.run()

    if response != Gtk.ResponseType.OK:
        dialog.destroy()
        return 1

    reading = dialog.reading_entry.get_text().strip()
    surface = dialog.surface_entry.get_text().strip()
    dialog.destroy()

    if not reading or not surface:
        return 1
    print(f"{reading}|{surface}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
