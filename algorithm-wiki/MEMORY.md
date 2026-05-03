# MEMORY — 算法 Wiki 索引

> 这是 LLM 的"记忆文件"。每次 ingest 新素材或新建条目后，这里会被自动追加。
> 人类可以浏览，但**不要手工编辑**——下一次 LLM 体检时会被覆盖。

---

## 原始素材索引（`raw/` 与 Vault）

### 来自 Vault（已有笔记，作为只读资料）

按主题汇总，详见 [`raw/from-vault.md`](raw/from-vault.md)：

- 题单：`Blind75.md` · `Grind75.md` · `Hot100.md` · `Leetcode150.md`
- 左程云课程：`算法笔记/zuoAlgorithm/`（含学习大纲、知识点总结、代码模板、960+ 题汇总）
- 专题笔记：`零茶山艾府+代码随想录/`（17 篇按题型专题）
- C++ 算法：`C++算法/`（入门笔记 + 必备知识点）
- labuladong：`labuladong/`（核心刷题框架等）
- 一些新题：`一些新题/`（左程云课程新增题目）

### 来自 raw/articles/ raw/papers/

- [Karpathy: LLM Knowledge Bases](raw/articles/karpathy-llm-knowledge-base.md) — 本 Wiki 方法论原帖

---

## Wiki 条目索引

### 概念 `wiki/concepts/`
- [[complexity]] — 时间复杂度 & 空间复杂度（大O、Master 定理、均摊）
- [[recursion]] — 递归思维（如何思考一个递归函数）
- [[divide-and-conquer]] — 分治范式

### 数据结构 `wiki/data-structures/`
- [[array]] — 数组（含动态数组 / 二维数组 / 滚动数组）
- [[linked-list]] — 链表（单 / 双 / 循环 / 哨兵）
- [[stack]] — 栈（顺序栈 / 链式栈 / 单调栈引子）
- [[queue]] — 队列（普通 / 双端 / 循环 / 单调队列引子）
- [[hash-table]] — 哈希表（开放寻址 / 拉链 / 常见用法）
- [[binary-tree]] — 二叉树（遍历 / 性质）
- [[bst]] — 二叉搜索树
- [[heap]] — 堆 / 优先队列
- [[union-find]] — 并查集
- [[graph]] — 图（存储 / 遍历 / 性质）
- [[trie]] — 字典树
- [[segment-tree]] — 线段树（含懒标记）
- [[fenwick-tree]] — 树状数组

### 算法 `wiki/algorithms/`
- [[sorting]] — 排序总览（10 大经典）
- [[binary-search]] — 二分查找（找数 / 找边界 / 浮点二分）
- [[bfs-dfs]] — BFS / DFS（树 + 图）
- [[topological-sort]] — 拓扑排序
- [[shortest-path]] — 最短路（Dijkstra / Bellman-Ford / SPFA / Floyd）
- [[mst]] — 最小生成树（Kruskal / Prim）
- [[kmp]] — KMP 字符串匹配
- [[manacher]] — Manacher 算法
- [[bit-manipulation]] — 位运算
- [[number-theory]] — 数论基础（GCD / 快速幂 / 素数筛 / 逆元）

### 解题技巧 `wiki/techniques/`
- [[two-pointers]] — 双指针（同向 / 相向 / 快慢）
- [[sliding-window]] — 滑动窗口（定长 / 不定长 / 多区间）
- [[prefix-sum]] — 前缀和 & 差分
- [[monotonic-stack]] — 单调栈
- [[monotonic-queue]] — 单调队列
- [[binary-search-answer]] — 二分答案
- [[discretization]] — 离散化
- [[scan-line]] — 扫描线

### 题型 `wiki/patterns/`
- [[backtracking]] — 回溯（子集 / 排列 / 组合 / 棋盘）
- [[greedy]] — 贪心（含证明思路）
- [[dp]] — 动态规划总论
- [[dp-knapsack]] — 背包 DP（01 / 完全 / 多重 / 分组）
- [[dp-interval]] — 区间 DP
- [[dp-tree]] — 树形 DP
- [[dp-bitmask]] — 状压 DP
- [[dp-digit]] — 数位 DP
- [[simulation]] — 模拟与构造

### 经典题 `wiki/problems/`
- [[problems/reverse-linked-list]] — 反转链表 (LC 206)
- [[problems/lru]] — LRU 缓存 (LC 146)
- [[problems/trapping-rain-water]] — 接雨水 (LC 42)
- [[problems/largest-rectangle]] — 柱状图最大矩形 (LC 84)
- [[problems/lis]] — 最长递增子序列 (LC 300)
- [[problems/edit-distance]] — 编辑距离 (LC 72)
- [[problems/word-ladder]] — 单词接龙 (LC 127)

### 术语表 `wiki/glossary/`
- [[terms]] — 中英术语对照表

### 概念地图 `wiki/maps/`
- [[algorithm-overview]] — 算法学习全景图
- [[ds-overview]] — 数据结构全景
- [[dp-roadmap]] — DP 学习路线图

---

## 待办（自动生成）

> 由 `lint.py` 或人类提问时新发现的"应存在但缺失"条目。

- [ ] `wiki/data-structures/avl-tree.md`（多次被引用）
- [ ] `wiki/data-structures/red-black-tree.md`
- [ ] `wiki/algorithms/string-hash.md`
- [ ] `wiki/algorithms/ac-automaton.md`
- [ ] `wiki/patterns/state-machine-dp.md`

---

## 最近一次 lint 结果

> 由 `scripts/lint.py` 写入，参见 `outputs/reports/lint-yyyy-mm-dd.md`。
