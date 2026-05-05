---
title: 模板说明
updated: 2026-05-05
---

# templates/ — Obsidian 模板库

## 📂 现有模板

| 模板 | 用途 | 何时用 | 推荐放到哪 |
|---|---|---|---|
| [`题目模板.md`](题目模板.md) | 新建单题文件 frontmatter + 骨架 | 每次刷新题 | `题目/0XXX-题名.md` |
| [`复习记录模板.md`](复习记录模板.md) | 记录复习一道题 / 一个概念 | 重做某题 / 复习专题时 | `outputs/reports/review-yyyy-mm-dd-XXX.md` |
| [`周报模板.md`](周报模板.md) | 每周刷题进度回顾（含 Dataview 查询） | 每周日 | `outputs/reports/yyyy-Wxx-周报.md` |
| [`综述模板.md`](综述模板.md) | 一个主题的全景总结 | 学完一个主题 / 面试前突击 | `outputs/reports/yyyy-mm-dd-{主题}-survey.md` |
| [`公司面经模板.md`](公司面经模板.md) | 一次面试的完整记录 | 面完一家公司 | `题单/公司Tag/{公司}/面经/yyyy-mm-dd-面经.md` |

## 🔧 怎么用（两种方式）

### 方式 A：Templater 插件（强烈推荐）⭐

**一次性安装**：

1. Obsidian → Settings → Community plugins → 搜 **Templater** → 安装 + 启用
2. Settings → Templater → "Template folder location" 设为 `templates`
3. Settings → Templater → "Hotkey for inserting template" 绑 `Option+Shift+T`
4. （可选）"Trigger Templater on new file creation" 开启 + 配置文件夹规则
   → 比如 `题目/` 自动套 `题目模板`，`outputs/reports/` 自动套 `周报模板`

**之后每次新建笔记**：

1. 在目标文件夹右键 → New note → 命名
2. 按 `Option+Shift+T` → 选要用的模板
3. frontmatter + 骨架自动填好，`{{date}}` 自动变今天

### 方式 B：手动复制粘贴（不装插件时用）

1. 打开模板文件 → `Cmd+A` 全选 → 复制
2. 在目标位置新建文件 → 粘贴
3. 把 `{{date}}` `{{title}}` 等手动替换

## 📋 模板字段速查

### `题目模板.md` — 字段含义

| 字段 | 说明 | 必填 |
|---|---|---|
| `leetcode` | LC 题号（数字） | ✅ |
| `title` | 题名 | ✅ |
| `url` | LC URL | ✅ |
| `difficulty` | `easy` / `medium` / `hard` | ✅ |
| `tags` | 一般化术语标签，如 `[linked-list, recursion]` | ✅ |
| `covers` | 对应 `algorithm-wiki/wiki/` 概念 slug | 推荐 |
| `sources` | 来源标识。新刷题用 `我的-自创`；多源题自动累加 | ✅ |
| `date_solved` | 做题日期，自动取今天 | ✅ |
| `solved` | `true` / `false` | ✅ |
| `time_to_solve` | 用时，如 `30min` | 可选 |
| `retry_needed` | 是否需重做（标 `true` 列入下次复习清单） | 可选 |
| `last_reviewed` | 上次复习日期（用于复习提醒） | 可选 |
| `kama` | 卡码网题号（替代 leetcode） | 仅卡码题 |
| `nowcoder` | 牛客题号 | 仅牛客题 |

### `covers` 字段可用的 slug（来自 `algorithm-wiki/wiki/`）

```
linked-list, binary-tree, bst, hash-table, heap, stack, queue, graph, trie,
union-find, segment-tree, fenwick-tree, array,
sorting, binary-search, bfs-dfs, topological-sort, shortest-path, mst, kmp,
manacher, bit-manipulation, number-theory,
two-pointers, sliding-window, prefix-sum, monotonic-stack, monotonic-queue,
binary-search-answer, discretization, scan-line,
backtracking, greedy, dp, dp-knapsack, dp-interval, dp-tree, dp-bitmask,
dp-digit, simulation
```

## 🎨 模板里的 Templater 变量

| 变量 | 渲染成 |
|---|---|
| `{{date}}` | 今天日期，如 `2026-05-05` |
| `{{date:YYYY-MM-DD}}` | 同上 |
| `{{date:YYYY-[W]WW}}` | 当前 ISO 周，如 `2026-W19` |
| `{{title}}` | 当前文件名（不含 `.md`） |
| `{{time}}` | 当前时间 |
| `<% tp.date.now("YYYY-MM-DD", -7) %>` | 7 天前日期（高级用法） |

## ➕ 想加新模板

1. 在本目录右键 → New note → 命名你的模板（如 `每日打卡模板`）
2. 写模板内容，用 `{{}}` 占位
3. 在本 README 「现有模板」表加一行
4. 保存即可。下次按 `Option+Shift+T` 就能选到

## 关联

- [`../SOP.md`](../SOP.md) — 仓库使用 SOP
- 详见 SOP 第「九、Obsidian 详细操作」章
