# outputs/ — 派生产物

由 `wiki/` 派生出来的东西放在这里：综述、对比表、刷题路线图、Marp 幻灯片、可视化图等。

## 子目录

```
outputs/
├── reports/        ← 综述、长答、对比表、lint 报告
└── slides/         ← Marp 幻灯片（.md，可在 Obsidian Marp 插件渲染）
```

## 命名约定

- **报告**：`reports/yyyy-mm-dd-题目简写.md`
- **幻灯片**：`slides/yyyy-mm-dd-题目简写.md`，文件首部应有 Marp frontmatter
- **lint 报告**：`reports/lint-yyyy-mm-dd.md`（由 `scripts/lint.py --write-report` 自动生成）

## Marp 幻灯片头模板

```markdown
---
marp: true
theme: default
size: 16:9
paginate: true
---

# 标题
副标题

---

## 第一页
...
```

VSCode + Marp 扩展或 Obsidian Marp 插件可以直接渲染。

## 工作流

1. 提问：用 LLM 基于 wiki 回答一个**复杂问题**
2. LLM 生成 `outputs/reports/...md` 或 `outputs/slides/...md`
3. 你读、修改、批注
4. 如果产物有持久价值，**回流**到 `wiki/` 中相应位置（例如新建一个条目）
5. 回流后该 outputs 文件可以保留作为 archive

## 已有产物

- [`reports/2026-05-03-monotonic-stack-vs-queue.md`](reports/2026-05-03-monotonic-stack-vs-queue.md) — 单调栈 vs 单调队列对比示例
