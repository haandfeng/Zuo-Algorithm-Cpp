#!/usr/bin/env python3
"""
build_index.py — 扫 题目/*.md，生成 题目/INDEX.md（按多种轴聚合）

输出：
  - 概览（统计：总数 / 难度分布 / 已做率 / 主题覆盖）
  - 按题号
  - 按难度
  - 按主题（covers）
  - 按来源（题单 / 课程）
  - 待办（缺 difficulty / 缺 tags 的题）

幂等：覆盖式重写 题目/INDEX.md
"""
from __future__ import annotations
import re
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
PROBLEMS = ROOT / "题目"

FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)


def parse_frontmatter(text: str) -> dict | None:
    m = FRONTMATTER_RE.match(text)
    if not m:
        return None
    fm: dict = {}
    current_key = None
    list_items: list[str] = []
    for line in m.group(1).split("\n"):
        if not line.strip():
            continue
        if line.startswith("  - "):
            if current_key is not None:
                list_items.append(line[4:].strip())
            continue
        if current_key is not None and list_items:
            fm[current_key] = list_items
            list_items = []
        kv = re.match(r"^([\w一-鿿_-]+):\s*(.*)$", line)
        if kv:
            key, val = kv.group(1), kv.group(2).strip()
            current_key = key
            list_items = []
            if val == "":
                fm[key] = None
            elif val == "[]":
                fm[key] = []
            elif val.startswith("[") and val.endswith("]"):
                inner = val[1:-1].strip()
                fm[key] = [x.strip() for x in inner.split(",") if x.strip()] if inner else []
            else:
                try:
                    fm[key] = int(val)
                except ValueError:
                    fm[key] = val
                current_key = None
    if current_key is not None and list_items:
        fm[current_key] = list_items
    return fm


def collect() -> list[dict]:
    out = []
    for f in sorted(PROBLEMS.glob("*.md")):
        if f.name in {"INDEX.md", "DASHBOARD.md", "README.md"}:
            continue
        text = f.read_text(encoding="utf-8")
        fm = parse_frontmatter(text)
        if not fm:
            continue
        fm["_path"] = f.relative_to(PROBLEMS).as_posix()
        fm["_basename"] = f.stem
        out.append(fm)
    return out


DIFF_ORDER = {"easy": 0, "medium": 1, "hard": 2, "": 3, None: 3}


def fmt_link(p: dict) -> str:
    title = p.get("title") or p["_basename"]
    return f"[[{p['_basename']}|{p.get('leetcode', '?')}. {title}]]"


def fmt_diff_badge(d: str | None) -> str:
    if d == "easy":   return "🟢"
    if d == "medium": return "🟡"
    if d == "hard":   return "🔴"
    return "⚪"


def render_index(problems: list[dict]) -> str:
    today = date.today().isoformat()
    n = len(problems)

    # 统计
    diff_counts = Counter(p.get("difficulty") or "未知" for p in problems)
    solved_counts = Counter("已做" if p.get("solved") else "未做" for p in problems)

    cover_to_problems: dict[str, list[dict]] = defaultdict(list)
    for p in problems:
        for c in p.get("covers") or []:
            cover_to_problems[c].append(p)

    source_to_problems: dict[str, list[dict]] = defaultdict(list)
    for p in problems:
        for s in p.get("sources") or []:
            source_to_problems[s].append(p)

    missing_diff = [p for p in problems if not p.get("difficulty")]
    missing_tags = [p for p in problems if not p.get("tags")]

    lines: list[str] = []
    lines.append("---")
    lines.append("title: 题目索引")
    lines.append(f"updated: {today}")
    lines.append("auto_generated: true")
    lines.append("---")
    lines.append("")
    lines.append(f"# 📚 题目索引（自动生成 · {today}）")
    lines.append("")
    lines.append(f"> 由 `algorithm-wiki/scripts/build_index.py` 扫描 `题目/*.md` 生成")
    lines.append(f"> **不要手动编辑**——重跑脚本即覆盖")
    lines.append("")

    # ============ 概览 ============
    lines.append("## 📊 概览")
    lines.append("")
    lines.append(f"- **总题数**：{n}")
    lines.append(f"- **难度分布**：🟢 easy {diff_counts.get('easy', 0)} · "
                 f"🟡 medium {diff_counts.get('medium', 0)} · "
                 f"🔴 hard {diff_counts.get('hard', 0)} · "
                 f"⚪ 未知 {diff_counts.get('未知', 0)}")
    lines.append(f"- **进度**：✅ 已做 {solved_counts.get('已做', 0)} · ⬜ 未做 {solved_counts.get('未做', 0)}")
    lines.append(f"- **概念覆盖**：{len(cover_to_problems)} 个 wiki 概念有对应题目")
    lines.append(f"- **来源数**：{len(source_to_problems)}")
    lines.append("")

    # ============ 跳转 ============
    lines.append("## 🔗 快速跳转")
    lines.append("")
    lines.append("- [按题号](#按题号)")
    lines.append("- [按难度](#按难度)")
    lines.append("- [按主题](#按主题)")
    lines.append("- [按来源](#按来源)")
    lines.append("- [待办（缺 frontmatter）](#待办)")
    lines.append("")

    # ============ 按题号 ============
    lines.append("## 按题号")
    lines.append("")
    lines.append("| LC | 难度 | 题名 | 主题 |")
    lines.append("|---:|:---:|---|---|")
    for p in sorted(problems, key=lambda x: (x.get("leetcode") or 99999)):
        lc = p.get("leetcode", "?")
        diff = fmt_diff_badge(p.get("difficulty"))
        title = p.get("title") or p["_basename"]
        link = f"[[{p['_basename']}|{title}]]"
        covers = ", ".join((p.get("covers") or [])[:3])
        lines.append(f"| {lc} | {diff} | {link} | {covers} |")
    lines.append("")

    # ============ 按难度 ============
    lines.append("## 按难度")
    lines.append("")
    for diff_label, diff_emoji in [("easy", "🟢"), ("medium", "🟡"), ("hard", "🔴"), (None, "⚪")]:
        sub = [p for p in problems if (p.get("difficulty") or None) == diff_label]
        if not sub:
            continue
        title = diff_label or "未知"
        lines.append(f"### {diff_emoji} {title}（{len(sub)} 题）")
        lines.append("")
        for p in sorted(sub, key=lambda x: (x.get("leetcode") or 99999)):
            lines.append(f"- {fmt_link(p)}")
        lines.append("")

    # ============ 按主题 ============
    lines.append("## 按主题")
    lines.append("")
    lines.append("> 基于 frontmatter 的 `covers` 字段，对应 algorithm-wiki/wiki/ 中的概念。")
    lines.append("")
    for cover in sorted(cover_to_problems.keys()):
        sub = cover_to_problems[cover]
        lines.append(f"### `{cover}`（{len(sub)} 题）")
        lines.append("")
        for p in sorted(sub, key=lambda x: (x.get("leetcode") or 99999)):
            diff = fmt_diff_badge(p.get("difficulty"))
            lines.append(f"- {diff} {fmt_link(p)}")
        lines.append("")

    # ============ 按来源 ============
    lines.append("## 按来源")
    lines.append("")
    for source in sorted(source_to_problems.keys()):
        sub = source_to_problems[source]
        lines.append(f"### `{source}`（{len(sub)} 题）")
        lines.append("")
        for p in sorted(sub, key=lambda x: (x.get("leetcode") or 99999)):
            diff = fmt_diff_badge(p.get("difficulty"))
            lines.append(f"- {diff} {fmt_link(p)}")
        lines.append("")

    # ============ 待办 ============
    lines.append("## 待办")
    lines.append("")
    lines.append(f"### 缺 difficulty（{len(missing_diff)} 题）")
    lines.append("")
    if missing_diff:
        lines.append("> 让 LLM 跑：")
        lines.append('> ```')
        lines.append('> 请扫描 题目/ 下 difficulty 为空的题目，根据 LeetCode 题号查难度，')
        lines.append('> 把结果写到 algorithm-wiki/data/lc_difficulty.json，然后重跑 fill_frontmatter.py')
        lines.append('> ```')
        lines.append("")
        for p in sorted(missing_diff, key=lambda x: (x.get("leetcode") or 99999)):
            lines.append(f"- {fmt_link(p)}")
        lines.append("")

    lines.append(f"### 缺 tags / covers（{len(missing_tags)} 题）")
    lines.append("")
    if missing_tags:
        lines.append("> 通常是只来自 题单/* 的题（题单不映射主题）。让 LLM 跑：")
        lines.append('> ```')
        lines.append('> 请扫描 题目/ 下 tags/covers 为空的题目，根据题目内容推断主题，')
        lines.append('> 直接写到对应 frontmatter 里')
        lines.append('> ```')
        lines.append("")
        for p in sorted(missing_tags, key=lambda x: (x.get("leetcode") or 99999))[:50]:
            lines.append(f"- {fmt_link(p)}")
        if len(missing_tags) > 50:
            lines.append(f"- ... 还有 {len(missing_tags) - 50} 题")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(f"_生成于 {today} · 用 `python algorithm-wiki/scripts/build_index.py` 重建_")
    return "\n".join(lines) + "\n"


def render_dashboard() -> str:
    """Dataview-based dashboard for users with the plugin."""
    return """---
title: 题目仪表板（Dataview）
type: dashboard
---

# 📊 题目仪表板

> 本页面使用 [Obsidian Dataview 插件](https://github.com/blacksmithgu/obsidian-dataview)。
> 如果你看到代码块没有渲染成表格 / 列表，去插件市场安装 Dataview。

## 总览（统计数）

```dataview
TABLE WITHOUT ID
  length(rows) as "总题数",
  length(filter(rows, (r) => r.difficulty = "easy")) as "🟢 easy",
  length(filter(rows, (r) => r.difficulty = "medium")) as "🟡 medium",
  length(filter(rows, (r) => r.difficulty = "hard")) as "🔴 hard",
  length(filter(rows, (r) => r.solved = true)) as "✅ 已做"
FROM "题目"
WHERE leetcode
GROUP BY true
```

## 按题号（全部）

```dataview
TABLE
  leetcode as "LC",
  difficulty as "难度",
  covers as "概念",
  sources as "来源"
FROM "题目"
WHERE leetcode
SORT leetcode ASC
```

## 按难度筛选 — 只看 Hard

```dataview
LIST
FROM "题目"
WHERE difficulty = "hard"
SORT leetcode ASC
```

## 按主题：动态规划

```dataview
LIST
FROM "题目"
WHERE contains(covers, "dp")
SORT leetcode ASC
```

## 按主题：链表

```dataview
LIST
FROM "题目"
WHERE contains(covers, "linked-list")
SORT leetcode ASC
```

## 按来源：Hot100

```dataview
LIST
FROM "题目"
WHERE contains(sources, "题单/Hot100")
SORT leetcode ASC
```

## 同时在 3+ 个来源里的"高频题"

```dataview
LIST
FROM "题目"
WHERE length(sources) >= 3
SORT length(sources) DESC, leetcode ASC
```

## 还没做的题

```dataview
LIST
FROM "题目"
WHERE solved != true
SORT leetcode ASC
```

## 缺 difficulty 的题（待 LLM 补）

```dataview
LIST
FROM "题目"
WHERE leetcode AND (difficulty = "" OR !difficulty)
SORT leetcode ASC
```

## 缺 covers 的题（只来自题单）

```dataview
LIST
FROM "题目"
WHERE leetcode AND (covers = [] OR !covers)
SORT leetcode ASC
```

---

## 自定义查询模板

复制这些代码块自己改：

### 按题号区间

```text
\\`\\`\\`dataview
LIST
FROM "题目"
WHERE leetcode >= 1 AND leetcode <= 100
SORT leetcode ASC
\\`\\`\\`
```

### 同时满足多个 covers

```text
\\`\\`\\`dataview
LIST
FROM "题目"
WHERE contains(covers, "binary-tree") AND contains(covers, "dp")
SORT leetcode ASC
\\`\\`\\`
```

### 按 last_reviewed 倒序（如果你给题加了这个字段）

```text
\\`\\`\\`dataview
TABLE difficulty, last_reviewed
FROM "题目"
WHERE last_reviewed
SORT last_reviewed DESC
\\`\\`\\`
```

---

## 跳到静态索引

如果你没装 Dataview 插件，看 [[INDEX|题目静态索引]]（每次重跑 build_index.py 重新生成）。
"""


def main() -> int:
    problems = collect()
    print(f"Collected {len(problems)} problems")

    index_md = render_index(problems)
    (PROBLEMS / "INDEX.md").write_text(index_md, encoding="utf-8")
    print(f"Wrote {(PROBLEMS / 'INDEX.md').relative_to(ROOT)} ({len(index_md)} chars)")

    dashboard_md = render_dashboard()
    (PROBLEMS / "DASHBOARD.md").write_text(dashboard_md, encoding="utf-8")
    print(f"Wrote {(PROBLEMS / 'DASHBOARD.md').relative_to(ROOT)} ({len(dashboard_md)} chars)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
