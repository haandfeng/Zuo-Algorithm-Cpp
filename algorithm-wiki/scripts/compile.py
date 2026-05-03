#!/usr/bin/env python3
"""
compile.py — emit a structured prompt that asks an LLM to compile new raw/
material into wiki/ articles.

This script does NOT call the LLM itself. Instead, it produces a prompt you
copy-paste into Claude Code / Cursor / ChatGPT, where the LLM has access to
the wiki and can read/write files.

Why this design?

The actual compilation requires:
  - Reading the raw source
  - Surveying existing wiki/ entries
  - Editing or creating multiple .md files
  - Preserving frontmatter, [[wikilinks]], references

That's exactly what an interactive coding agent (Claude Code) does best.
This script just packages the request, including the relevant file paths
and the wiki conventions, so you can hand it off in one paste.

Usage:
    python compile.py <raw_path>            # print prompt to stdout
    python compile.py <raw_path> --copy     # also copy to clipboard (mac: pbcopy)

Example:
    python compile.py raw/articles/2026-04-12-bfs-vs-dfs.md
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW = ROOT / "WORKFLOW.md"
MEMORY = ROOT / "MEMORY.md"


def render_prompt(raw_path: str) -> str:
    return dedent(f"""
        # Task: Compile new source into the algorithm wiki

        ## Source
        Read: `{raw_path}`

        ## Conventions
        Follow the protocol described in `algorithm-wiki/WORKFLOW.md`:

        1. Read the source file
        2. Survey relevant existing articles in `algorithm-wiki/wiki/`
           (use `python algorithm-wiki/scripts/search.py "关键词"` if needed)
        3. Identify 2~10 candidate concepts. For each:
           - **EXISTING**: open the article and add the new fact in the right
             section, then append the source to the `参考资料` block.
           - **NEW**: create `algorithm-wiki/wiki/<subdir>/<slug>.md` using
             the article template at the bottom of `WORKFLOW.md`.
        4. Each new article must have ≥ 2 outbound `[[wikilink]]`s.
        5. If a new MAJOR topic appears, add a node to the relevant
           `algorithm-wiki/wiki/maps/*.md`.
        6. Append the source to `## 原始素材索引` in
           `algorithm-wiki/MEMORY.md` (or use `ingest.py` first if not done).

        ## Style
        - 中文正文 + 英文术语 + C++ 代码示例为主
        - 每个事实尽量带 `参考资料` 引用回 raw/ 或 Vault 笔记
        - 文章长度 600~3000 字；过长就拆分

        ## Output
        After making the edits, give a one-screen summary:
        - 新建的条目（路径 + 一句话描述）
        - 更新的条目（路径 + 改了哪一节）
        - 待用户决定的歧义点（如果有）

        Then propose 3 follow-up questions the user might want to explore.
    """).strip() + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate an LLM prompt to compile a raw source into wiki.")
    ap.add_argument("raw_path", help="Path to the raw source (relative to algorithm-wiki/).")
    ap.add_argument("--copy", action="store_true", help="Copy to clipboard (mac: pbcopy).")
    args = ap.parse_args()

    target = Path(args.raw_path)
    abs_target = target if target.is_absolute() else (ROOT / target)
    if not abs_target.exists():
        print(f"[warn] source file not found at expected location: {abs_target}", file=sys.stderr)
        print("[warn] continuing — the prompt will still be generated.", file=sys.stderr)

    prompt = render_prompt(args.raw_path)
    print(prompt)

    if args.copy:
        if shutil.which("pbcopy"):
            subprocess.run(["pbcopy"], input=prompt, text=True, check=True)
            print("[ok] copied to clipboard (pbcopy)", file=sys.stderr)
        elif shutil.which("xclip"):
            subprocess.run(["xclip", "-selection", "clipboard"], input=prompt, text=True, check=True)
            print("[ok] copied to clipboard (xclip)", file=sys.stderr)
        else:
            print("[warn] no clipboard tool found (pbcopy / xclip)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
