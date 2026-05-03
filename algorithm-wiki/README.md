# 算法 Wiki

> 一个由 LLM 自动维护、人工只读的**结构化算法知识库**
> 方法论来源：[Andrej Karpathy 的 LLM Knowledge Bases 实践](raw/articles/karpathy-llm-knowledge-base.md)
> 服务对象：你（左程云课程 + LeetCode Hot100/Blind75/Grind75 准备 + 北美 SDE 面试）

## 这是什么

这是一个**关于算法 / 数据结构 / 解题范式**的中文 Wiki，建立在你已有的 Obsidian Vault 之上：

- **`raw/`**：原始素材索引，主要指回 Vault 里你已有的笔记（题单、专题笔记、左程云课程笔记）
- **`wiki/`**：由 LLM 阅读 `raw/` 后"编译"出的**结构化概念网络**，每条概念一篇短文、互相 `[[wikilink]]`
- **`scripts/`**：辅助脚本（ingest、lint、search）
- **`outputs/`**：基于 wiki 派生的产物（综述、对比表、Marp 幻灯片、可视化图）

人类是**产品经理 + 提问者**：刷题、贴新题进 `raw/`、提问、读 wiki。
LLM 是**研究员 + 编辑**：读、写、整理、自检、回答。

## 与你现有笔记的关系

你的 Vault 已经按 **题单（Hot100/Blind75/Grind75/LC150）** 和 **题型专题（labuladong / 代码随想录 / 灵神）** 组织。这套组织方式适合**做题流程**，但不适合**回答"我对这个概念到底懂不懂"的快速查阅**。

本 Wiki 补的就是这个空缺：

| 已有笔记（你写的） | 本 Wiki（LLM 写的） |
|---|---|
| 按题单 / 课程节次组织 | 按**概念**组织 |
| 每个文件 = 一道题 / 一节课 | 每个文件 = 一个**概念** |
| 内容深、长 | 短小、互链密、可索引 |
| 你做笔记 | LLM 自动维护 |
| 学习现场 | 概念地图 |

它们**互相引用**：本 Wiki 的"参考资料"段会大量指向你 Vault 里的具体题目和章节。

## 快速上手

```bash
# 1. 用 Obsidian 打开整个左程云算法/ 这个 Vault（你已经在用）
#    然后从侧栏进入 algorithm-wiki/wiki/index.md

# 2. ingest 一道刚刷完的新题：
python scripts/ingest.py "../一些新题/最长递增子序列.md"

# 3. 问 LLM 一个跨主题的问题（在 Claude Code / Cursor 中）：
#    "请基于 wiki/ 给我整理：单调栈和单调队列的本质区别 + 各自适用场景，配 3 道代表题"
#    LLM 会读 wiki/techniques/ 和 wiki/patterns/ 下的相关条目，并参考你已有题目笔记给出答案

# 4. 体检：找断链、孤立节点、信息矛盾
python scripts/lint.py

# 5. 在 wiki 中搜索关键词
python scripts/search.py "单调栈"
```

## 目录结构

```
algorithm-wiki/
├── README.md                       ← 你正在读的文件
├── WORKFLOW.md                     ← LLM 操作手册
├── MEMORY.md                       ← LLM 维护的索引
│
├── raw/                            ← 原始素材
│   ├── README.md
│   ├── from-vault.md               ← 指回 Vault 已有笔记的索引
│   ├── articles/                   ← Web Clipper 抓取
│   ├── papers/                     ← 论文 / 摘录
│   └── images/                     ← 截图 / 图示
│
├── wiki/                           ← 编译产物（短小互联）
│   ├── index.md                    ← 主入口
│   ├── maps/                       ← 概念地图（MOC）
│   ├── concepts/                   ← 复杂度、递归、分治
│   ├── data-structures/            ← 数组/链表/栈/树/图/堆/Trie/并查集...
│   ├── algorithms/                 ← 排序/二分/搜索/最短路/字符串
│   ├── techniques/                 ← 双指针/滑动窗口/单调栈/前缀和/二分答案
│   ├── patterns/                   ← 回溯/贪心/DP（背包/区间/树形/状压/数位）
│   ├── problems/                   ← 经典题（反转链表/LRU/接雨水/...）
│   └── glossary/                   ← 中英术语表
│
├── scripts/                        ← 工具
│   ├── ingest.py                   ← 把新素材登记进 MEMORY.md
│   ├── compile.py                  ← 用 LLM 把 raw/ 编译进 wiki/
│   ├── lint.py                     ← 体检：断链 / 孤立 / 矛盾
│   └── search.py                   ← 简易关键词搜索
│
└── outputs/                        ← 派生产物
    ├── slides/                     ← Marp 幻灯片
    └── reports/                    ← 综述 / 对比表 / 长答
```

## 阅读路线（推荐）

第一次进来，按这个顺序看：

1. [算法学习全景图](wiki/maps/algorithm-overview.md) ← 总览
2. [复杂度分析](wiki/concepts/complexity.md) → [递归](wiki/concepts/recursion.md) → [分治](wiki/concepts/divide-and-conquer.md)
3. **数据结构**：[数组](wiki/data-structures/array.md) → [链表](wiki/data-structures/linked-list.md) → [栈与队列](wiki/data-structures/stack.md) → [哈希表](wiki/data-structures/hash-table.md) → [二叉树](wiki/data-structures/binary-tree.md) → [堆](wiki/data-structures/heap.md) → [并查集](wiki/data-structures/union-find.md) → [图](wiki/data-structures/graph.md)
4. **解题技巧**：[双指针](wiki/techniques/two-pointers.md) → [滑动窗口](wiki/techniques/sliding-window.md) → [前缀和](wiki/techniques/prefix-sum.md) → [单调栈](wiki/techniques/monotonic-stack.md) → [二分答案](wiki/techniques/binary-search-answer.md)
5. **核心范式**：[二分查找](wiki/algorithms/binary-search.md) → [BFS / DFS](wiki/algorithms/bfs-dfs.md) → [回溯](wiki/patterns/backtracking.md) → [贪心](wiki/patterns/greedy.md) → [动态规划](wiki/patterns/dp.md)
6. **进阶**：[最短路](wiki/algorithms/shortest-path.md) · [KMP](wiki/algorithms/kmp.md) · [背包 DP](wiki/patterns/dp-knapsack.md)

或者直接打开 [wiki/index.md](wiki/index.md) 自由跳转。

## 设计原则（给 LLM 阅读）

1. **单一信息源**：同一概念只在一处展开，其它地方只 `[[wikilink]]`
2. **小而互联**：宁可写 5 篇 1500 字短文加密集互链，也不写 1 篇大长文
3. **代码示例**：以 **C++** 为主语言（与你 Vault 的主语言一致），必要时给 Python 对照
4. **强追溯**：每个事实 / 模板尽量在文末 `参考资料` 指回 `raw/` 或 Vault 里的具体题目
5. **频繁体检**：每次大规模 ingest 后跑 `lint.py`，修孤立节点、断链
6. **保持中文**：除算法/题目英文名、类名、变量名、命令外，正文一律简体中文

## Obsidian 集成提示

- 整个 Vault 你已经用 Obsidian 打开了；`algorithm-wiki/` 是其中一个普通子目录
- 推荐结合插件：
  - **Graph view**：看 wiki 内部的链接图谱
  - **Backlinks pane**：每篇 wiki 文章右侧自动显示哪些题目笔记引用了它
  - **Excalidraw**：你已装；用于画手绘图存到 `raw/images/`
  - **Marp** / **Slides**：把 `outputs/slides/*.md` 渲染成幻灯片
  - **Dataview**：用 frontmatter 写复杂查询

## 致谢

方法论原帖参见 [`raw/articles/karpathy-llm-knowledge-base.md`](raw/articles/karpathy-llm-knowledge-base.md)。
左程云课程见 [B 站频道](https://space.bilibili.com/8888480)。
