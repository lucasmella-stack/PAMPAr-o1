#!/usr/bin/env python3
"""Add SPDX headers to Python source files.

This script prepends SPDX license identifier to .py files under src/ and llarri_o1/.
It avoids changing files that already contain an SPDX-License-Identifier line.

It preserves shebang and encoding headers when present.

Usage:
  python scripts/add_spdx_headers.py
"""

from __future__ import annotations

from pathlib import Path

SPDX_LINE = "# SPDX-License-Identifier: AGPL-3.0-or-later\n"
COPYRIGHT_LINE = "# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi / Segunda Cabeza\n"


def _has_spdx(text: str) -> bool:
    return "SPDX-License-Identifier:" in text


def _insert_header(original: str) -> str:
    lines = original.splitlines(keepends=True)

    insert_at = 0

    # Keep shebang
    if lines and lines[0].startswith("#!"):
        insert_at = 1

    # Keep encoding cookie in first or second line
    if len(lines) > insert_at and ("coding" in lines[insert_at] or "encoding" in lines[insert_at]):
        insert_at += 1

    header = SPDX_LINE + COPYRIGHT_LINE

    # Avoid double blank lines
    if insert_at < len(lines) and lines[insert_at].strip() == "":
        return "".join(lines[:insert_at] + [header] + lines[insert_at:])

    return "".join(lines[:insert_at] + [header, "\n"] + lines[insert_at:])


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    targets = [repo_root / "src", repo_root / "llarri_o1"]

    changed = 0
    scanned = 0

    for base in targets:
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if path.name.startswith("__pycache__"):
                continue
            scanned += 1
            text = path.read_text(encoding="utf-8")
            if _has_spdx(text):
                continue
            path.write_text(_insert_header(text), encoding="utf-8")
            changed += 1

    print(f"Scanned: {scanned} files")
    print(f"Updated: {changed} files")


if __name__ == "__main__":
    main()
