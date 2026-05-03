#!/usr/bin/env python3
"""
ingest.py — register a new raw source into MEMORY.md and stub a wiki article if needed.

Usage:
    python ingest.py <path_to_raw_or_vault_note> [--title "..."] [--summary "..."]

What it does:
    1. Computes a relative path from algorithm-wiki/ to the source file.
    2. Reads the file, extracts a candidate title (first H1 or filename).
    3. Appends an entry to MEMORY.md under "## 原始素材索引".
    4. Prints a checklist of next steps for the LLM to run (which wiki entries
       to create / update). The actual wiki compilation is left to the LLM.

This script is intentionally simple — it doesn't call any LLM. The "compile"
step is `compile.py`, which is an LLM-driven action you run from your editor.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent       # .../algorithm-wiki
MEMORY = ROOT / "MEMORY.md"
INDEX_HEADER = "## 原始素材索引"


def first_h1(text: str) -> str | None:
    for line in text.splitlines():
        m = re.match(r"^#\s+(.+?)\s*$", line)
        if m:
            return m.group(1)
    return None


def first_paragraph(text: str, max_len: int = 100) -> str:
    # Skip frontmatter and headings, find first non-empty line of prose.
    in_fm = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("---"):
            in_fm = not in_fm
            continue
        if in_fm or not line or line.startswith("#") or line.startswith(">"):
            continue
        return (line[:max_len] + "…") if len(line) > max_len else line
    return ""


def relpath_from_root(target: Path) -> str:
    """Return a path relative to algorithm-wiki/, even if target is outside it."""
    try:
        return target.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        # Outside of ROOT: build with ../ jumps
        try:
            from os.path import relpath
            return relpath(target.resolve(), ROOT)
        except Exception:
            return str(target.resolve())


def update_memory(rel: str, title: str, summary: str) -> bool:
    """Append `- [title](rel) — summary` under the index header. Returns True if added."""
    if not MEMORY.exists():
        print(f"[error] {MEMORY} not found", file=sys.stderr)
        return False
    text = MEMORY.read_text(encoding="utf-8")
    entry_line = f"- [{title}]({rel}) — {summary}"

    if entry_line in text:
        print(f"[skip] entry already present: {entry_line}")
        return False

    if INDEX_HEADER not in text:
        print(f"[error] '{INDEX_HEADER}' section not found in MEMORY.md", file=sys.stderr)
        return False

    # Insert under the header (after the next blank line).
    out_lines: list[str] = []
    inserted = False
    in_section = False
    for line in text.splitlines(keepends=True):
        out_lines.append(line)
        if not inserted and line.startswith(INDEX_HEADER):
            in_section = True
            continue
        # Insert before the next top-level header (## ...) once we're past the index section.
        if in_section and not inserted and line.startswith("## ") and not line.startswith(INDEX_HEADER):
            # back-pop the new header, insert, then append
            out_lines.pop()
            out_lines.append(entry_line + "\n\n")
            out_lines.append(line)
            inserted = True

    if not inserted:
        # Section was at end of file
        out_lines.append("\n" + entry_line + "\n")

    MEMORY.write_text("".join(out_lines), encoding="utf-8")
    print(f"[ok] appended to MEMORY.md: {entry_line}")
    return True


def suggest_next_steps(rel: str, title: str) -> None:
    print()
    print("─" * 60)
    print("Next steps for the LLM (run via Claude Code / Cursor):")
    print("─" * 60)
    print(f"""
请阅读 `{rel}` 并完成以下 ingest 流程（参考 WORKFLOW.md）：

1. 在 wiki/ 中识别 2~10 个需要更新或新建的概念条目
2. 对已有条目，把新事实写入合适小节，并在「参考资料」段加来源
3. 对应当新建的概念，在合适子目录新建 .md，遵循文章模板
4. 给每个新条目加至少 2 个 [[wikilink]] 形成反向链接
5. 如果是一级主题，更新 wiki/maps/ 下的相应 MOC
6. 完成后输出简短报告：
   - 更新的条目
   - 新建的条目
   - 待人类确认的歧义

引用建议：在文章末尾的「参考资料」段加入：
    - [[{rel}]]
""")


def main() -> int:
    ap = argparse.ArgumentParser(description="Register a raw source into the algorithm wiki.")
    ap.add_argument("path", help="Path to the raw file (md/txt/pdf/...) — relative or absolute.")
    ap.add_argument("--title", help="Override the title (default: first H1 or filename).")
    ap.add_argument("--summary", help="Override the one-line summary (default: first paragraph).")
    args = ap.parse_args()

    target = Path(args.path)
    if not target.exists():
        # Allow path relative to ROOT or its parent (the Vault root)
        for base in (ROOT, ROOT.parent):
            candidate = base / args.path
            if candidate.exists():
                target = candidate
                break
        else:
            print(f"[error] file not found: {args.path}", file=sys.stderr)
            return 1

    text = ""
    if target.suffix.lower() in {".md", ".txt"}:
        try:
            text = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = target.read_text(encoding="utf-8", errors="replace")

    title = args.title or first_h1(text) or target.stem
    summary = args.summary or first_paragraph(text) or "(待补充摘要)"

    rel = relpath_from_root(target)
    print(f"[info] source = {rel}")
    print(f"[info] title  = {title}")
    print(f"[info] summary= {summary}")
    print(f"[info] today  = {date.today().isoformat()}")

    update_memory(rel, title, summary)
    suggest_next_steps(rel, title)
    return 0


if __name__ == "__main__":
    sys.exit(main())
