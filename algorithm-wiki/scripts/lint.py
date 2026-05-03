#!/usr/bin/env python3
"""
lint.py — health check on the algorithm wiki.

Checks performed:
  1. Broken [[wikilinks]]: link targets not found.
  2. Orphan articles: no incoming wikilinks (excluding index/MOC).
  3. Missing-but-referenced: linked-to but no .md exists.
  4. Missing frontmatter: no `---` block at top of file.
  5. Missing 参考资料 / 关联条目 section.
  6. Stub articles: < 200 chars body.

Usage:
    python lint.py                  # human-readable report
    python lint.py --json           # machine-readable
    python lint.py --write-report   # writes outputs/reports/lint-YYYY-MM-DD.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent       # .../algorithm-wiki
WIKI = ROOT / "wiki"
REPORTS = ROOT / "outputs" / "reports"
VAULT_ROOT = ROOT.parent                              # the Obsidian vault root
# Extra dirs in the vault that wiki articles legitimately link into via [[basename]]
EXTRA_DIRS = ["课程", "题单", "题目", "语言"]

WIKILINK = re.compile(r"\[\[([^\[\]#]+?)(?:#[^\]\|]*)?(?:\\?\|[^\]]*)?\]\]")
FRONTMATTER = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def all_md_files() -> list[Path]:
    return sorted(p for p in WIKI.rglob("*.md") if p.is_file())


def slug_index(files: list[Path]) -> dict[str, Path]:
    """Map basename-without-ext (and full path slug) to file path."""
    idx: dict[str, Path] = {}
    for f in files:
        idx[f.stem] = f                             # short slug
        idx[f.relative_to(WIKI).with_suffix("").as_posix()] = f   # path slug
    # Also index vault-wide files so cross-links to 课程/题单/题目/语言 don't show as broken
    for sub in EXTRA_DIRS:
        d = VAULT_ROOT / sub
        if not d.exists():
            continue
        for f in d.rglob("*.md"):
            if any(part.startswith(".") for part in f.parts):
                continue
            # Only register stem if not already taken by a wiki article
            if f.stem not in idx:
                idx[f.stem] = f
    return idx


def parse_links(text: str) -> set[str]:
    return {m.group(1).strip() for m in WIKILINK.finditer(text)}


def is_index_or_moc(rel_path: str) -> bool:
    return rel_path.startswith("maps/") or rel_path == "index.md" or rel_path.endswith("README.md")


def lint() -> dict[str, Any]:
    files = all_md_files()
    idx = slug_index(files)

    broken: list[tuple[str, str]] = []        # (file_path, link_target)
    missing: dict[str, list[str]] = defaultdict(list)   # target -> referers
    backlinks: dict[Path, set[str]] = defaultdict(set)  # file -> referers (slug)
    no_frontmatter: list[str] = []
    no_refs_section: list[str] = []
    stubs: list[tuple[str, int]] = []

    for f in files:
        text = f.read_text(encoding="utf-8")
        rel = f.relative_to(WIKI).as_posix()

        # frontmatter
        if not FRONTMATTER.match(text):
            no_frontmatter.append(rel)

        # references section
        if "参考资料" not in text and "关联条目" not in text:
            no_refs_section.append(rel)

        # stub
        body = FRONTMATTER.sub("", text)
        if len(body.strip()) < 200:
            stubs.append((rel, len(body.strip())))

        # links
        for target in parse_links(text):
            target = target.strip().rstrip("\\")
            # External (vault-relative) link starting with ../ — verify on disk, don't count as broken
            if target.startswith("../") or target.startswith("/"):
                resolved = (f.parent / target).resolve()
                if resolved.exists() or resolved.with_suffix(".md").exists():
                    pass   # valid cross-vault link, ignore
                # If file doesn't exist we still don't report — vault layout outside our scope
                continue
            normalized = target.split("/")[-1]
            if normalized in idx:
                target_file = idx[normalized]
                backlinks[target_file].add(rel)
            elif target in idx:
                backlinks[idx[target]].add(rel)
            else:
                broken.append((rel, target))
                missing[target].append(rel)

    orphans: list[str] = []
    for f in files:
        rel = f.relative_to(WIKI).as_posix()
        if is_index_or_moc(rel):
            continue
        if not backlinks.get(f):
            orphans.append(rel)

    return {
        "files_count": len(files),
        "broken_links": broken,
        "missing_targets": dict(missing),
        "orphans": orphans,
        "no_frontmatter": no_frontmatter,
        "no_refs_section": no_refs_section,
        "stubs": stubs,
    }


def render_human(report: dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Lint Report — {date.today().isoformat()}")
    lines.append("")
    lines.append(f"- 文件总数：{report['files_count']}")
    lines.append(f"- 断链：{len(report['broken_links'])}")
    lines.append(f"- 孤立条目：{len(report['orphans'])}")
    lines.append(f"- 缺 frontmatter：{len(report['no_frontmatter'])}")
    lines.append(f"- 缺参考/关联段：{len(report['no_refs_section'])}")
    lines.append(f"- 桩文章（< 200 字符）：{len(report['stubs'])}")
    lines.append("")

    if report["broken_links"]:
        lines.append("## 🔗 断链")
        for src, tgt in report["broken_links"][:50]:
            lines.append(f"- `{src}` → [[{tgt}]]")
        if len(report["broken_links"]) > 50:
            lines.append(f"- ... 还有 {len(report['broken_links']) - 50} 条")
        lines.append("")

    if report["missing_targets"]:
        lines.append("## ❓ 被引用但缺失的条目（建议新建）")
        for tgt, refs in sorted(report["missing_targets"].items(),
                                 key=lambda kv: -len(kv[1])):
            lines.append(f"- `{tgt}` —— 被 {len(refs)} 处引用")
        lines.append("")

    if report["orphans"]:
        lines.append("## 🏝 孤立条目（无人引用）")
        for o in report["orphans"]:
            lines.append(f"- `{o}`")
        lines.append("")

    if report["no_frontmatter"]:
        lines.append("## 📄 缺 frontmatter")
        for f in report["no_frontmatter"]:
            lines.append(f"- `{f}`")
        lines.append("")

    if report["stubs"]:
        lines.append("## 🌱 桩文章")
        for f, n in report["stubs"]:
            lines.append(f"- `{f}` ({n} 字)")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Lint the algorithm wiki.")
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--write-report", action="store_true",
                    help="write to outputs/reports/lint-YYYY-MM-DD.md")
    args = ap.parse_args()

    report = lint()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    text = render_human(report)
    print(text)

    if args.write_report:
        REPORTS.mkdir(parents=True, exist_ok=True)
        out = REPORTS / f"lint-{date.today().isoformat()}.md"
        out.write_text(text, encoding="utf-8")
        print(f"\n[ok] report written to {out.relative_to(ROOT)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
