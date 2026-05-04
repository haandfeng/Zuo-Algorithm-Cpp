---
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
\`\`\`dataview
LIST
FROM "题目"
WHERE leetcode >= 1 AND leetcode <= 100
SORT leetcode ASC
\`\`\`
```

### 同时满足多个 covers

```text
\`\`\`dataview
LIST
FROM "题目"
WHERE contains(covers, "binary-tree") AND contains(covers, "dp")
SORT leetcode ASC
\`\`\`
```

### 按 last_reviewed 倒序（如果你给题加了这个字段）

```text
\`\`\`dataview
TABLE difficulty, last_reviewed
FROM "题目"
WHERE last_reviewed
SORT last_reviewed DESC
\`\`\`
```

---

## 跳到静态索引

如果你没装 Dataview 插件，看 [[INDEX|题目静态索引]]（每次重跑 build_index.py 重新生成）。
