---
title: 模板说明
updated: 2026-05-05
---

# templates/ — Obsidian 模板

## 文件

| 模板 | 用途 |
|---|---|
| [`题目模板.md`](题目模板.md) | 新建 `题目/0XXX-题名.md` 时一键填充 frontmatter |

## 怎么用（两种方式）

### 方式 A：Templater 插件（推荐）⭐

**一次性安装**：

1. Obsidian → Settings → Community plugins → 搜 **Templater** → 安装 + 启用
2. Settings → Templater → "Template folder location" 设为 `templates`
3. （可选）Settings → Templater → "Hotkey" 给 "Insert template" 绑定 `Cmd+Shift+T`

**之后每次新题**：

1. 在 `题目/` 文件夹右键 → New note
2. 命名：`0752-打开转盘锁`（4 位题号 + 中划线 + 题名）
3. 按 `Cmd+Shift+T`（或命令面板 → "Templater: Insert template"）
4. 选 `题目模板`
5. frontmatter 自动填好 + `{{date}}` 自动变成今天日期
6. 直接编辑题号、题名、URL、解法即可

### 方式 B：手动复制粘贴

1. 打开 [`题目模板.md`](题目模板.md)
2. 选中全部内容（`Cmd+A`）→ 复制
3. 在 `题目/` 新建文件 → 粘贴
4. 把 `{{date}}` 手动改成今天日期

## 模板字段说明

| 字段 | 说明 | 必填 |
|---|---|---|
| `leetcode` | LC 题号（数字） | ✅ |
| `title` | 题名 | ✅ |
| `url` | LC URL | ✅ |
| `difficulty` | `easy` / `medium` / `hard` | ✅ |
| `tags` | 一般化术语标签，如 `[linked-list, recursion]` | ✅ |
| `covers` | 对应 `algorithm-wiki/wiki/` 概念 slug，如 `[bfs-dfs, graph]` | 推荐 |
| `sources` | 来源标识。新刷题用 `我的-自创`；多源题会自动累加 | ✅ |
| `date_solved` | 做题日期，自动取今天 | ✅ |
| `solved` | `true` 或 `false` | ✅ |
| `time_to_solve` | 用时，如 `30min` | 可选 |
| `retry_needed` | 是否需重做（标 `true` 列入下次复习清单） | 可选 |
| `last_reviewed` | 上次复习日期（用于复习提醒） | 可选 |
| `kama` | 卡码网题号（替代 leetcode 字段） | 仅卡码题 |
| `nowcoder` | 牛客题号 | 仅牛客题 |

## covers 字段如何选

打开 `algorithm-wiki/wiki/index.md`，看里面有哪些概念 slug。常见的：

```
linked-list, binary-tree, bst, hash-table, heap, stack, queue, graph, trie,
union-find, segment-tree, fenwick-tree, array,
sorting, binary-search, bfs-dfs, topological-sort, shortest-path, mst, kmp,
manacher, bit-manipulation, number-theory,
two-pointers, sliding-window, prefix-sum, monotonic-stack, monotonic-queue,
binary-search-answer, discretization, scan-line,
backtracking, greedy, dp, dp-knapsack, dp-interval, dp-tree, dp-bitmask, dp-digit, simulation
```

## 关联

- [`SOP.md`](../SOP.md) - 仓库使用 SOP（场景 2）
