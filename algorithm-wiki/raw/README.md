# raw/ — 原始素材

这里存放所有未经 LLM 整理的**原始资料**。

```
raw/
├── README.md           ← 你正在读的文件
├── from-vault.md       ← 指回上层 Vault 已有笔记的索引
├── articles/           ← 博客 / 推文 / Web Clipper 抓取
├── papers/             ← 论文 / PDF 摘录
└── images/             ← 截图 / 图示
```

## 三种素材来源

### 1. Vault 已有笔记（最大头）

你已经写了大量左程云课程笔记 + 题解。它们都**留在原位**，本 Wiki 通过 [`from-vault.md`](from-vault.md) 索引指过去。**不要把它们移到 raw/articles/**——那会破坏 Vault 既有结构。

### 2. 外部抓取

新读到的算法博客、Codeforces 编辑、官方解答等：
- 用 Obsidian Web Clipper 保存到 `raw/articles/yyyy-mm-dd-作者-标题简写.md`
- 顶部加 frontmatter：`source: <URL>`、`captured: yyyy-mm-dd`

### 3. 论文 / PDF

- 算法论文（如 KMP 原论文、Tarjan 的并查集论文）放 `raw/papers/`
- PDF 旁附 `.notes.md` 写自己的笔记

## 投放规则

1. **文件名**：`yyyy-mm-dd-作者-标题简写.md`，便于排序
2. **保留元数据**：顶部 frontmatter 至少有 `source` + `captured`
3. **不要修改正文**：保留原文以便后续比对；自己的看法用 `> 编者按：…` 引用块
4. **图片本地化**：把文章中引用的图片下载到 `raw/images/<文章 slug>/`，链接改成本地路径
5. **PDF 旁加 notes**：`<论文名>.pdf` + `<论文名>.notes.md`

## 与 wiki/ 的关系

- `raw/` + `from-vault.md` 是**输入**，`wiki/` 是**输出**
- 永远只增不删（除非确认是垃圾或重复）
- 每条 `raw/` 素材应当至少被一个 `wiki/` 条目在 `参考资料` 中引用，否则就是孤立素材

## 已收录素材

参见 [`../MEMORY.md`](../MEMORY.md) 的「原始素材索引」一节（自动维护）。
