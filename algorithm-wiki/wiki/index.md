---
title: 算法 Wiki 主索引
tags: [index, hub]
updated: 2026-05-03
---

# 算法 Wiki 主索引

> 这是 Wiki 的中央入口。所有条目按主题组织如下。
> 想看可视化关系网，请打开 [[algorithm-overview|算法学习全景图]]。

## 🧱 基础概念 `concepts/`

| 条目 | 一句话 |
|---|---|
| [[complexity]] | 时间 / 空间复杂度，大 O、Master 定理、均摊 |
| [[recursion]] | 递归"如何思考"：定义子问题、写出递推 |
| [[divide-and-conquer]] | 分治三段式：拆 / 解 / 合 |

## 🗃 数据结构 `data-structures/`

| 条目 | 关键词 |
|---|---|
| [[array]] | 数组、动态数组、二维 / 滚动 |
| [[linked-list]] | 单 / 双 / 循环 / 哨兵 |
| [[stack]] | 顺序 / 链式 / 单调栈引子 |
| [[queue]] | 普通 / 双端 / 循环 / 单调队列引子 |
| [[hash-table]] | 拉链 / 开放寻址 / 常见用法 |
| [[binary-tree]] | 遍历、性质、递推 |
| [[bst]] | 二叉搜索树（含有序集合） |
| [[heap]] | 堆 / 优先队列（Top-K、调度） |
| [[union-find]] | 并查集（路径压缩 + 按秩合并） |
| [[graph]] | 邻接表 / 邻接矩阵 / 性质 |
| [[trie]] | 字典树（前缀计数、最大异或对） |
| [[segment-tree]] | 线段树（懒标记） |
| [[fenwick-tree]] | 树状数组 |

## ⚙️ 算法 `algorithms/`

| 条目 | 关键词 |
|---|---|
| [[sorting]] | 10 大排序总览 |
| [[binary-search]] | 找数 / 找边界 / 浮点 |
| [[bfs-dfs]] | 树 / 图遍历的两种范式 |
| [[topological-sort]] | 拓扑排序（Kahn / DFS） |
| [[shortest-path]] | Dijkstra / Bellman-Ford / SPFA / Floyd |
| [[mst]] | Kruskal / Prim |
| [[kmp]] | KMP 字符串匹配 |
| [[manacher]] | 马拉车（最长回文子串 O(N)） |
| [[bit-manipulation]] | 位运算技巧 |
| [[number-theory]] | GCD、快速幂、素数筛、逆元 |

## 🔧 解题技巧 `techniques/`

| 条目 | 信号词 |
|---|---|
| [[two-pointers]] | "有序 / 双侧扫描 / 快慢" |
| [[sliding-window]] | "连续子数组 / 子串 + 满足某约束" |
| [[prefix-sum]] | "区间求和 / 区间是否…" |
| [[monotonic-stack]] | "下一个更大 / 更小元素" |
| [[monotonic-queue]] | "滑窗内最大 / 最小" |
| [[binary-search-answer]] | "求最小的最大 / 最大的最小" |
| [[discretization]] | 数据范围大但值数少 |
| [[scan-line]] | 区间合并 / 矩形面积 |

## 🎯 题型范式 `patterns/`

| 条目 | 关键词 |
|---|---|
| [[backtracking]] | 子集 / 排列 / 组合 / 棋盘 |
| [[greedy]] | 局部最优 → 全局最优；含证明 |
| [[dp]] | 动态规划总论：状态 / 转移 / 边界 |
| [[dp-knapsack]] | 01 / 完全 / 多重 / 分组背包 |
| [[dp-interval]] | 区间 DP（戳气球、合并石头） |
| [[dp-tree]] | 树形 DP（打家劫舍 III、最大独立集） |
| [[dp-bitmask]] | 状压 DP（旅行商、棋盘） |
| [[dp-digit]] | 数位 DP |
| [[simulation]] | 模拟与构造 |

## 🧪 经典题 `problems/`

按领域分类，方便面试前突击：

- 链表：[[problems/reverse-linked-list]] · [[problems/lru]]
- 单调栈：[[problems/trapping-rain-water]] · [[problems/largest-rectangle]]
- DP：[[problems/lis]] · [[problems/edit-distance]]
- BFS/DFS：[[problems/word-ladder]]

## 📖 术语表

[[terms|中英术语对照]]

## 🗺 概念地图

- [[algorithm-overview]] — 算法学习全景图
- [[ds-overview]] — 数据结构全景
- [[dp-roadmap]] — DP 学习路线
