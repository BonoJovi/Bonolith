#!/usr/bin/env python3
"""Interactive curation tool for Bonolith conversion eval dataset.

Reads captured conversions from $HOME/.local/share/bonolith/conversions.jsonl
(produced when Bonolith runs with BONOLITH_LOG_CONVERSIONS=1) and lets you mark
mis-conversions as eval cases, appended to tests/conversion_cases/cases.jsonl.

Usage:
    python3 scripts/curate_cases.py
    python3 scripts/curate_cases.py --only-user-selected   # show only entries where
                                                            # user re-selected at least one segment
                                                            # (likely-wrong signal)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_PATH = REPO_ROOT / "tests" / "conversion_cases" / "cases.jsonl"
LOG_PATH = Path(os.path.expanduser("~/.local/share/bonolith/conversions.jsonl"))
CURSOR_PATH = Path(os.path.expanduser("~/.local/share/bonolith/curation_cursor"))

CATEGORIES = ["segmentation", "word_choice", "inflection", "both"]
POS_SOLVABLE = ["yes", "partial", "no"]


def load_log() -> list[dict]:
    if not LOG_PATH.exists():
        sys.exit(f"No capture log at {LOG_PATH}. Run Bonolith with BONOLITH_LOG_CONVERSIONS=1 first.")
    out = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def read_cursor() -> int:
    if CURSOR_PATH.exists():
        try:
            return int(CURSOR_PATH.read_text().strip())
        except ValueError:
            return 0
    return 0


def write_cursor(ts: int) -> None:
    CURSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    CURSOR_PATH.write_text(str(ts))


def next_case_id() -> str:
    if not EVAL_PATH.exists():
        return "case_0001"
    last_num = 0
    for line in EVAL_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cid = json.loads(line).get("id", "")
            if cid.startswith("case_"):
                last_num = max(last_num, int(cid[5:]))
        except (json.JSONDecodeError, ValueError):
            continue
    return f"case_{last_num + 1:04d}"


def prompt(label: str, default: str = "", choices: list[str] | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    if choices:
        suffix = f" ({'/'.join(choices)}){suffix}"
    while True:
        val = input(f"{label}{suffix}: ").strip()
        if not val and default:
            val = default
        if not val:
            continue
        if choices and val not in choices:
            print(f"  -> must be one of {choices}")
            continue
        return val


def curate_entry(entry: dict) -> dict | None:
    print("\n" + "=" * 60)
    print(f"kana:      {entry.get('kana', '')}")
    print(f"composed:  {entry.get('composed', '')}")
    print(f"timestamp: {dt.datetime.fromtimestamp(entry.get('ts', 0))}")
    print("segments:")
    for i, seg in enumerate(entry.get("segments", [])):
        mark = "*" if seg.get("user_selected") else " "
        alts = seg.get("alternatives", [])[:5]
        print(f"  {mark} [{i}] {seg.get('reading')} -> {seg.get('selected')}  alts={alts}")
    print("  (* = user re-selected this segment)")

    action = prompt("Action", default="s", choices=["a", "s", "q"])
    if action == "q":
        return "QUIT"  # sentinel
    if action == "s":
        return None

    expected_str = prompt('expected (space-separated segments, e.g. "今日は いい 天気")')
    expected = expected_str.split()
    category = prompt("category", default="word_choice", choices=CATEGORIES)
    subcategory = prompt("subcategory (free text, optional)", default="-")
    pos_solvable = prompt("pos_solvable", default="partial", choices=POS_SOLVABLE)
    pos_hypothesis = prompt("pos_hypothesis (free text, optional)", default="-")
    notes = prompt("notes (free text, optional)", default="-")

    return {
        "id": next_case_id(),
        "input_hiragana": entry.get("kana", ""),
        "expected": expected,
        "actual": [s.get("selected", "") for s in entry.get("segments", [])],
        "category": category,
        "subcategory": "" if subcategory == "-" else subcategory,
        "pos_solvable": pos_solvable,
        "pos_hypothesis": "" if pos_hypothesis == "-" else pos_hypothesis,
        "notes": "" if notes == "-" else notes,
        "date_collected": dt.date.today().isoformat(),
        "bonolith_version": entry.get("version", ""),
        "source": "captured",
    }


def append_case(case: dict) -> None:
    EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with EVAL_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(case, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-user-selected", action="store_true",
                    help="only show entries where the user re-selected at least one segment")
    args = ap.parse_args()

    entries = load_log()
    cursor = read_cursor()
    new_entries = [e for e in entries if e.get("ts", 0) > cursor]

    if args.only_user_selected:
        new_entries = [e for e in new_entries
                       if any(s.get("user_selected") for s in e.get("segments", []))]

    if not new_entries:
        print("No new entries since last curation. (cursor:", cursor, ")")
        return

    print(f"{len(new_entries)} entry/entries to review. a=add case, s=skip, q=quit")
    last_ts = cursor
    for entry in new_entries:
        result = curate_entry(entry)
        if result == "QUIT":
            break
        if result is not None:
            append_case(result)
            print(f"  -> appended {result['id']}")
        last_ts = max(last_ts, entry.get("ts", 0))

    write_cursor(last_ts)
    print(f"\nCursor advanced to ts={last_ts}.")


if __name__ == "__main__":
    main()
