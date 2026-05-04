---
title: 公司面试题集
updated: 2026-05-04
---

# 公司Tag — 公司面试题集

按公司整理面试题。每家公司一个子目录，独立维护。

## 推荐目录组织

每家公司里建议这样组织：

```
<公司名>/
├── README.md           # 公司面试特点、常考主题、最近热门
├── 高频/               # 这家公司近期 Top N 高频题
│   ├── 001-题名.md
│   └── ...
├── 面经/               # 面经收录（按时间 / 岗位 / 部门）
│   └── 2026-XX-面经.md
└── 笔记.md             # 整体观察、tips、coding style 偏好
```

每道题文件统一格式（参考 [`USAGE.md`](../../USAGE.md) 的"场景 1"）：

```yaml
---
leetcode: 206
company: Meta
difficulty: medium
date_seen: 2026-05-04
covers: [linked-list, recursion]
solved: true
---
```

## 已有公司

| 公司 | 状态 | 备注 |
|---|---|---|
| **TikTok** | 已建立 | 你正在准备的目标公司 |

## 计划新增

| 公司 | 优先级 | 重点考察 |
|---|---|---|
| Meta | ⭐⭐⭐ | 系统设计 + 高频算法 + 行为面 |
| Google | ⭐⭐⭐ | 算法深度 + 数据结构 + 设计 |
| Amazon | ⭐⭐⭐ | 14 条 LP + 中等算法 |
| Apple | ⭐⭐ | C++ 偏多、底层 |
| Microsoft | ⭐⭐ | 经典题 + 系统设计 |
| Bloomberg | ⭐⭐ | OOD + STL + C++ |
| Snowflake / Databricks | ⭐ | 数据 / 分布式 |

## 添加新公司的步骤

1. 在本目录创建 `<公司名>/`
2. 在该公司目录加 `README.md`（公司简介 + 面试特点）
3. 把本文档「已有公司」表格更新一行
4. （可选）在 [`algorithm-wiki/MEMORY.md`](../../algorithm-wiki/MEMORY.md) 里也登记一笔

## 资源建议

- LeetCode 公司题库：付费会员能直接看公司 tag
- **CSOAHelp** / **一亩三分地** / **小红书**：搜面经
- LinkedIn 同岗位的人总结的笔记

## 关联

- 算法概念：[`../../algorithm-wiki/wiki/index.md`](../../algorithm-wiki/wiki/index.md)
- 通用题单：[`../Hot100.md`](../Hot100.md) / [`../Blind75.md`](../Blind75.md) / [`../Grind75.md`](../Grind75.md) / [`../Leetcode150.md`](../Leetcode150.md)
- 单题深度笔记：[`../../题目/`](../../题目/)
