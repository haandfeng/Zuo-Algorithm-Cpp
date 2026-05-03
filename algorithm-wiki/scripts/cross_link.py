#!/usr/bin/env python3
"""
Phase 2: 自动给 algorithm-wiki/wiki/ 下每篇文章注入「我 Vault 里的相关笔记」段。

策略：
  - 给每个 wiki 概念定义一组关键词（中文 + 英文 alias）
  - 扫描 课程/, 题单/, 题目/ 下所有 .md 文件，按关键词在文件名中匹配
  - 按来源分组（灵神+Carl / 左程云 / labuladong / C++算法 / 题单 / 题目）
  - 用 [[basename|显示文本]] 格式注入，让 Obsidian 能解析

幂等：再跑一次会替换已有的「我 Vault 里的相关笔记」段。
"""

from __future__ import annotations
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path("/Users/ha/Documents/study/左程云算法")
WIKI = ROOT / "algorithm-wiki" / "wiki"

SECTION_HEADER = "## 📂 我 Vault 里的相关笔记"
SECTION_RE = re.compile(
    r"\n## 📂 我 Vault 里的相关笔记\n.*?(?=\n## |\Z)",
    re.DOTALL,
)

# Keyword map: wiki article basename -> (display title, [keywords])
# 关键词在文件名中做大小写不敏感的子串匹配
ARTICLES = {
    # 概念
    "complexity":           ("复杂度",   ["复杂度", "Big-O", "大O", "Master"]),
    "recursion":            ("递归",     ["递归"]),
    "divide-and-conquer":   ("分治",     ["分治"]),
    # 数据结构
    "array":                ("数组",     ["数组"]),
    "linked-list":          ("链表",     ["链表"]),
    "stack":                ("栈",       ["栈"]),
    "queue":                ("队列",     ["队列"]),
    "hash-table":           ("哈希表",   ["哈希", "hash"]),
    "binary-tree":          ("二叉树",   ["二叉树"]),
    "bst":                  ("BST",      ["二叉搜索", "BST", "搜索树"]),
    "heap":                 ("堆",       ["堆", "优先队列"]),
    "union-find":           ("并查集",   ["并查集"]),
    "graph":                ("图",       ["图"]),
    "trie":                 ("Trie",     ["Trie", "字典树", "前缀树"]),
    "segment-tree":         ("线段树",   ["线段树"]),
    "fenwick-tree":         ("树状数组", ["树状数组", "Fenwick", "BIT"]),
    # 算法
    "sorting":              ("排序",     ["排序", "快排", "归并", "堆排", "基数"]),
    "binary-search":        ("二分查找", ["二分查找", "二分"]),
    "bfs-dfs":              ("BFS/DFS",  ["BFS", "DFS", "广度", "深度", "搜索"]),
    "topological-sort":     ("拓扑排序", ["拓扑"]),
    "shortest-path":        ("最短路",   ["最短路", "Dijkstra", "Floyd"]),
    "mst":                  ("最小生成树", ["最小生成树", "Kruskal", "Prim", "MST"]),
    "kmp":                  ("KMP",      ["KMP"]),
    "manacher":             ("Manacher", ["Manacher", "回文"]),
    "bit-manipulation":     ("位运算",   ["位运算", "二进制", "异或"]),
    "number-theory":        ("数论",     ["数论", "GCD", "素数", "快速幂"]),
    # 技巧
    "two-pointers":         ("双指针",   ["双指针"]),
    "sliding-window":       ("滑动窗口", ["滑动窗口", "滑窗"]),
    "prefix-sum":           ("前缀和",   ["前缀和", "差分"]),
    "monotonic-stack":      ("单调栈",   ["单调栈"]),
    "monotonic-queue":      ("单调队列", ["单调队列"]),
    "binary-search-answer": ("二分答案", ["二分答案"]),
    "discretization":       ("离散化",   ["离散化"]),
    "scan-line":            ("扫描线",   ["扫描线"]),
    # 范式
    "backtracking":         ("回溯",     ["回溯", "暴力搜索", "DFS"]),
    "greedy":               ("贪心",     ["贪心"]),
    "dp":                   ("动态规划", ["动态规划", "DP"]),
    "dp-knapsack":          ("背包DP",   ["背包"]),
    "dp-interval":          ("区间DP",   ["区间DP", "区间 DP"]),
    "dp-tree":              ("树形DP",   ["树形DP", "树形 DP"]),
    "dp-bitmask":           ("状压DP",   ["状压", "位掩码", "bitmask"]),
    "dp-digit":             ("数位DP",   ["数位DP", "数位 DP"]),
    "simulation":           ("模拟",     ["模拟", "构造"]),
}

# 文件扫描的根目录组（用于分组展示）
GROUPS = [
    ("灵茶山艾府 + 代码随想录", ROOT / "课程" / "灵茶山艾府 + 代码随想录"),
    ("左程云",     ROOT / "课程" / "左程云"),
    ("labuladong", ROOT / "课程" / "labuladong"),
    ("C++算法",    ROOT / "课程" / "C++算法"),
    ("题单",       ROOT / "题单"),
    ("题目",       ROOT / "题目"),
]

MAX_LINKS_PER_GROUP = 6     # 每组最多展示几条匹配


def catalog() -> dict[str, list[Path]]:
    """For each group label, list all .md files (excluding hidden/system)."""
    out: dict[str, list[Path]] = {}
    for label, root in GROUPS:
        if not root.exists():
            out[label] = []
            continue
        files = []
        for f in root.rglob("*.md"):
            # skip hidden / .obsidian etc.
            if any(part.startswith(".") for part in f.parts):
                continue
            files.append(f)
        out[label] = sorted(files)
    return out


def matches(file: Path, keywords: list[str]) -> bool:
    """Match if any keyword appears in filename or any path segment."""
    name = file.stem  # 文件名（不带 .md）
    target = name.lower()
    for kw in keywords:
        if kw.lower() in target:
            return True
    return False


def display_link(f: Path) -> str:
    """Render an Obsidian wikilink with full path as alt text for human readability."""
    rel = f.relative_to(ROOT).as_posix()
    return f"[[{f.stem}|{rel}]]"


def build_section(article_slug: str, all_files: dict[str, list[Path]]) -> str | None:
    """Build the cross-link section for one wiki article. Returns None if no matches."""
    if article_slug not in ARTICLES:
        return None
    title, keywords = ARTICLES[article_slug]

    blocks: list[tuple[str, list[Path]]] = []
    total = 0
    for label, files in all_files.items():
        hits = [f for f in files if matches(f, keywords)]
        if hits:
            blocks.append((label, hits[:MAX_LINKS_PER_GROUP]))
            total += len(hits[:MAX_LINKS_PER_GROUP])

    if total == 0:
        return None

    lines = [SECTION_HEADER]
    lines.append("")
    lines.append(f"> 自动扫描 Vault 中含「{title}」相关关键词（{', '.join(keywords)}）的笔记，按来源分组。")
    lines.append("")
    for label, hits in blocks:
        lines.append(f"### {label}")
        for h in hits:
            lines.append(f"- {display_link(h)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def inject(md_file: Path, section: str) -> bool:
    """Inject (or replace) the section right before the last '## 关联条目' or '## 参考资料', or append at end."""
    text = md_file.read_text(encoding="utf-8")

    # If section already exists, replace it
    if SECTION_HEADER in text:
        new_text = SECTION_RE.sub("\n" + section, text, count=1)
        if new_text != text:
            md_file.write_text(new_text, encoding="utf-8")
            return True
        return False

    # Otherwise insert before "## 参考资料" (preferred) or "## 关联条目"
    insert_marker = None
    for marker in ("## 参考资料", "## 关联条目"):
        if marker in text:
            insert_marker = marker
            break

    if insert_marker:
        idx = text.index(insert_marker)
        new_text = text[:idx] + section + "\n" + text[idx:]
    else:
        # Append at end
        new_text = text.rstrip() + "\n\n" + section

    md_file.write_text(new_text, encoding="utf-8")
    return True


def main() -> int:
    all_files = catalog()

    print("=== Catalog ===")
    for label, files in all_files.items():
        print(f"  {label}: {len(files)} files")
    print()

    injected = 0
    skipped = 0
    for slug in ARTICLES:
        # Find the wiki article (could be in concepts/, data-structures/, etc.)
        candidates = list(WIKI.rglob(f"{slug}.md"))
        if not candidates:
            print(f"  [skip] no article found for {slug}.md")
            skipped += 1
            continue
        if len(candidates) > 1:
            print(f"  [warn] multiple articles for {slug}.md: {[c.name for c in candidates]}")
        md_file = candidates[0]
        section = build_section(slug, all_files)
        if section is None:
            print(f"  [skip] no matches for {slug}")
            skipped += 1
            continue
        ok = inject(md_file, section)
        if ok:
            injected += 1
            print(f"  [ok] {md_file.relative_to(ROOT)}")

    print()
    print(f"Done. Injected/updated {injected} articles. Skipped {skipped}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
