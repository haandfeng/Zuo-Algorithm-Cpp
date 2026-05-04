#!/usr/bin/env python3
"""
拆主题文件 → 单题文件。

用法：
    python split_topic.py <topic_file> [<topic_file> ...]
    python split_topic.py --dry-run <topic_file>     # 只打印不动文件

策略：
    - 识别 `## [题号. 题名](leetcode-url)` 形式的标题作为题目边界
    - 提取每个题目的内容（标题之间）
    - 写入 题目/0题号-题名.md，加上 frontmatter
    - 同一题已存在 → 追加新「来源」段，不覆盖
    - 主题文件被重写为索引（每题指向 题目/）

可逆：git 全程跟踪，不满意 git checkout 即可。
"""

from __future__ import annotations
import argparse
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent  # vault root
PROBLEM_DIR = ROOT / "题目"

# 匹配 `## [123. 题名](https://leetcode.cn/problems/xxx/)` 形式
PROBLEM_HEADING = re.compile(
    r"^##\s+\[(\d+)[\.\s]+(.+?)\]\((https?://[^)]+)\)\s*$",
    re.MULTILINE,
)

# 文件名安全字符（保留中英数字 + 中划线 + 中文）
def safe_name(s: str) -> str:
    # Replace forbidden chars in macOS / git filenames
    s = s.strip()
    for ch in r'\/:*?"<>|':
        s = s.replace(ch, " ")
    s = s.replace("  ", " ").strip()
    return s

def build_frontmatter(num: int, name: str, url: str, source_label: str, source_topic: str) -> str:
    return (
        f"---\n"
        f"leetcode: {num}\n"
        f"title: {name}\n"
        f"url: {url}\n"
        f"difficulty: \n"        # 留空，让用户/LLM 后续填
        f"tags: []\n"
        f"covers: []\n"
        f"sources:\n"
        f"  - {source_label}/{source_topic}\n"
        f"date_split: {date.today().isoformat()}\n"
        f"solved: true\n"
        f"---\n\n"
    )


def build_problem_md(num: int, name: str, url: str, body: str,
                     source_label: str, source_topic: str) -> str:
    fm = build_frontmatter(num, name, url, source_label, source_topic)
    h1 = f"# {num}. {name}\n\n[LeetCode 链接]({url})\n\n"
    src_block = f"## 来源：{source_label} / {source_topic}\n\n{body.strip()}\n"
    return fm + h1 + src_block


def append_source(existing_path: Path, num: int, name: str, body: str,
                  source_label: str, source_topic: str) -> bool:
    """If problem file exists, append a new source section. Returns True if appended."""
    text = existing_path.read_text(encoding="utf-8")
    new_section_header = f"## 来源：{source_label} / {source_topic}"

    # Skip if this source is already in the file
    if new_section_header in text:
        return False

    # Update sources list in frontmatter
    fm_match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    if fm_match:
        fm_block = fm_match.group(1)
        if "sources:" in fm_block:
            new_fm_block = re.sub(
                r"(sources:\s*\n(?:  - .+\n)*)",
                lambda m: m.group(1) + f"  - {source_label}/{source_topic}\n",
                fm_block,
                count=1,
            )
        else:
            new_fm_block = fm_block + f"\nsources:\n  - {source_label}/{source_topic}"
        text = "---\n" + new_fm_block + "\n---\n" + text[fm_match.end():]

    new_text = text.rstrip() + "\n\n" + new_section_header + "\n\n" + body.strip() + "\n"
    existing_path.write_text(new_text, encoding="utf-8")
    return True


def split_one(topic_file: Path, source_label: str, dry_run: bool = False) -> dict:
    """Split one topic file. Returns stats dict."""
    text = topic_file.read_text(encoding="utf-8")
    matches = list(PROBLEM_HEADING.finditer(text))

    stats = {
        "topic_file": str(topic_file.relative_to(ROOT)),
        "problems_found": len(matches),
        "created": [],
        "appended": [],
        "skipped": [],
    }

    if not matches:
        stats["skipped"].append("no problems found")
        return stats

    source_topic = topic_file.stem  # e.g., "链表"

    # Header content (before first problem)
    header_content = text[:matches[0].start()].rstrip()

    # Index lines (will replace topic file)
    index_lines = []
    if header_content:
        index_lines.append(header_content)
        index_lines.append("")
    index_lines.append("---")
    index_lines.append("")
    index_lines.append(f"## 📑 题目索引（自动拆分到 [`题目/`](../../题目/)）")
    index_lines.append("")
    index_lines.append(f"> 本主题原始内容已按题号拆分到 `题目/` 下单题文件。")
    index_lines.append(f"> 拆分日期：{date.today().isoformat()}，来源标签：`{source_label}/{source_topic}`")
    index_lines.append("")

    PROBLEM_DIR.mkdir(parents=True, exist_ok=True)

    for i, m in enumerate(matches):
        num = int(m.group(1))
        name = safe_name(m.group(2))
        url = m.group(3).strip()

        section_start = m.start()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        # Body = section minus the heading line
        section = text[section_start:section_end]
        section_body = "\n".join(section.split("\n", 1)[1:]).strip()

        slug = f"{num:04d}-{name}"
        problem_path = PROBLEM_DIR / f"{slug}.md"

        if problem_path.exists():
            # Append source section
            if dry_run:
                stats["appended"].append(slug + " [DRY]")
            else:
                appended = append_source(problem_path, num, name, section_body, source_label, source_topic)
                if appended:
                    stats["appended"].append(slug)
                else:
                    stats["skipped"].append(slug + " (source already present)")
        else:
            # Create new
            content = build_problem_md(num, name, url, section_body, source_label, source_topic)
            if dry_run:
                stats["created"].append(slug + " [DRY]")
            else:
                problem_path.write_text(content, encoding="utf-8")
                stats["created"].append(slug)

        index_lines.append(f"- LC {num} [{name}]({url}) → [[{slug}|题目/{slug}]]")

    # Rewrite topic file as index
    if not dry_run:
        new_text = "\n".join(index_lines).rstrip() + "\n"
        topic_file.write_text(new_text, encoding="utf-8")
        stats["index_rewritten"] = True
    else:
        stats["index_rewritten"] = False

    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Split topic files into per-problem files.")
    ap.add_argument("paths", nargs="+", help="Topic file(s) to split")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--source-label", default="auto",
                    help="Source label (default: derive from parent folder)")
    args = ap.parse_args()

    total_created = 0
    total_appended = 0
    for p in args.paths:
        topic_file = Path(p) if Path(p).is_absolute() else (ROOT / p)
        if not topic_file.exists():
            print(f"[error] not found: {topic_file}", file=sys.stderr)
            continue

        # Derive source_label from parent folder if 'auto'
        if args.source_label == "auto":
            label = topic_file.parent.name
        else:
            label = args.source_label

        stats = split_one(topic_file, label, dry_run=args.dry_run)

        print(f"\n=== {stats['topic_file']} ===")
        print(f"  Problems found: {stats['problems_found']}")
        print(f"  Created:  {len(stats['created'])}")
        for s in stats["created"][:5]:
            print(f"    + {s}")
        if len(stats["created"]) > 5:
            print(f"    + ... +{len(stats['created']) - 5} more")
        print(f"  Appended: {len(stats['appended'])}")
        for s in stats["appended"][:5]:
            print(f"    ~ {s}")
        if len(stats["appended"]) > 5:
            print(f"    ~ ... +{len(stats['appended']) - 5} more")
        print(f"  Skipped:  {len(stats['skipped'])}")
        for s in stats["skipped"][:3]:
            print(f"    - {s}")

        total_created += len(stats["created"])
        total_appended += len(stats["appended"])

    print(f"\n{'[DRY RUN] ' if args.dry_run else ''}Total: {total_created} created, {total_appended} appended.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
