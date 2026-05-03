---
title: 算法术语对照表
aliases: [Glossary, Terms, 术语表]
tags: [glossary, reference]
created: 2026-05-03
updated: 2026-05-03
---

# 算法术语对照表

> 中英对照 + 一句话定义 + 指向 Wiki 中的详细条目。
> 按字母 / 拼音排序。

## A

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| AC 自动机 | Aho-Corasick Automaton | Trie 上做 KMP，多模式串匹配 | — |
| 邻接表 | Adjacency List | 用列表存图中每个节点的出边 | [[../data-structures/graph]] |
| 邻接矩阵 | Adjacency Matrix | 用 N×N 矩阵存图 | [[../data-structures/graph]] |
| 算法 | Algorithm | 解决某个问题的有限步骤 | — |
| 摊还 | Amortized | "单步可能慢，但 N 步合起来便宜" | [[../concepts/complexity]] |
| 数组 | Array | 连续内存的线性表 | [[../data-structures/array]] |

## B

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| 回溯 | Backtracking | DFS + 撤销选择 | [[../patterns/backtracking]] |
| 大 O | Big O | 渐近上界 | [[../concepts/complexity]] |
| 二分查找 | Binary Search | 有序序列对半查 | [[../algorithms/binary-search]] |
| 二分答案 | Binary Search the Answer | 二分最优解 | [[../techniques/binary-search-answer]] |
| 二叉树 | Binary Tree | 每节点最多 2 子 | [[../data-structures/binary-tree]] |
| 二叉搜索树 | BST, Binary Search Tree | 左 < 根 < 右 | [[../data-structures/bst]] |
| 位运算 | Bit Manipulation | 直接操作二进制位 | [[../algorithms/bit-manipulation]] |
| 位掩码 | Bitmask | 用整数位表示集合 | [[../patterns/dp-bitmask]] |
| 广度优先搜索 | BFS, Breadth-First Search | 用队列逐层扩展 | [[../algorithms/bfs-dfs]] |
| BPE | Byte-Pair Encoding | （词法学，非算法）— |
| 桶排序 | Bucket Sort | 按值域分桶 | [[../algorithms/sorting]] |

## C

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| 卡塔兰数 | Catalan Number | 1, 1, 2, 5, 14, ... 计数公式 | — |
| 调度场算法 | Shunting Yard | 中缀转后缀 | [[../data-structures/stack]] |
| 字符集 | Character Set | 字符可能出现的范围 | — |
| 组合 | Combination | 选 K 个不计顺序 | [[../patterns/backtracking]] |
| 完全二叉树 | Complete Binary Tree | 除最后一层全满 + 最后一层左对齐 | [[../data-structures/binary-tree]] |
| 复杂度 | Complexity | 时间 / 空间 | [[../concepts/complexity]] |
| 计数排序 | Counting Sort | 值域小的桶排序变种 | [[../algorithms/sorting]] |
| CoT | Chain of Thought | （LLM 概念，非算法） |

## D

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| DAG | Directed Acyclic Graph | 有向无环图 | [[../algorithms/topological-sort]] |
| 双端队列 | Deque | 两端可 push/pop | [[../data-structures/queue]] |
| 深度优先搜索 | DFS, Depth-First Search | 用栈/递归深入到底 | [[../algorithms/bfs-dfs]] |
| 差分 | Difference Array | 前缀和的逆运算 | [[../techniques/prefix-sum]] |
| Dijkstra | — | 单源非负权最短路 | [[../algorithms/shortest-path]] |
| 离散化 | Discretization | 把大值域映射到小下标 | — |
| 分治 | Divide and Conquer | 分→治→合 | [[../concepts/divide-and-conquer]] |
| 动态规划 | DP, Dynamic Programming | 重叠子问题 + 记忆 | [[../patterns/dp]] |
| 双指针 | Two Pointers | 两个指针协同扫描 | [[../techniques/two-pointers]] |

## E

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| 编辑距离 | Edit Distance, Levenshtein | 把 A 变 B 的最少操作 | [[../problems/edit-distance]] |
| Embedding | Embedding | （ML 概念）— |
| 涌现能力 | Emergent Abilities | （LLM 概念）— |
| 欧拉路径 | Eulerian Path | 经过每条边一次 | — |
| 异或 | XOR | `^` | [[../algorithms/bit-manipulation]] |

## F

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| 树状数组 | Fenwick Tree, BIT | 动态前缀和 | [[../data-structures/fenwick-tree]] |
| FlashAttention | — | （LLM 概念）— |
| Floyd | — | 多源最短路 O(N³) | [[../algorithms/shortest-path]] |
| 函数 | Function | 输入输出映射 | — |

## G

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| 图 | Graph | 节点 + 边 | [[../data-structures/graph]] |
| 贪心 | Greedy | 每步局部最优 | [[../patterns/greedy]] |
| GCD | Greatest Common Divisor | 最大公约数 | — |

## H

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| 哈希 | Hash | 把 key 映射为整数 | [[../data-structures/hash-table]] |
| 哈希表 | Hash Table / Hash Map | hash + 桶 | [[../data-structures/hash-table]] |
| 堆 | Heap | 完全二叉树 + 父子大小关系 | [[../data-structures/heap]] |
| 堆排序 | Heap Sort | 用堆排序 | [[../algorithms/sorting]] |

## I

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| 入度 | In-degree | 指向节点的边数 | [[../algorithms/topological-sort]] |
| 插入排序 | Insertion Sort | 每次插到正确位置 | [[../algorithms/sorting]] |
| 区间 | Interval | [l, r] 一段 | [[../patterns/dp-interval]] |

## J - K

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| 跳表 | Skip List | 多层索引链表 | — |
| KMP | Knuth-Morris-Pratt | 字符串匹配 O(N+M) | [[../algorithms/kmp]] |
| 背包 | Knapsack | 容量 + 价值最优 | [[../patterns/dp-knapsack]] |
| Kruskal | — | 最小生成树 | [[../algorithms/mst]] |

## L

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| LCS | Longest Common Subsequence | 最长公共子序列 | [[../patterns/dp]] |
| LIS | Longest Increasing Subsequence | 最长递增子序列 | [[../problems/lis]] |
| 链式前向星 | Chained Forward Star | 紧凑的图存储 | [[../data-structures/graph]] |
| 链表 | Linked List | 节点 + 指针 | [[../data-structures/linked-list]] |
| 后进先出 | LIFO | 栈 | [[../data-structures/stack]] |
| LRU | Least Recently Used | 淘汰最久未用 | [[../problems/lru]] |
| LCM | Least Common Multiple | 最小公倍数 | — |

## M

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| Master 定理 | Master Theorem | 解分治递推 | [[../concepts/complexity]] |
| 记忆化 | Memoization | DP 备忘录 | [[../patterns/dp]] |
| 归并排序 | Merge Sort | 分治排序 | [[../algorithms/sorting]] |
| 最小生成树 | MST | 连通无环 + 总权最小 | [[../algorithms/mst]] |
| MoE | Mixture of Experts | （LLM 概念）— |
| 单调栈 | Monotonic Stack | 栈中保持单调 | [[../techniques/monotonic-stack]] |
| 单调队列 | Monotonic Queue | 滑窗最值 | [[../techniques/monotonic-queue]] |

## N - O

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| 网格 | Grid | 二维数组当图 | [[../data-structures/graph]] |
| 节点 | Node / Vertex | 图的"点" | [[../data-structures/graph]] |
| 出度 | Out-degree | 节点出去的边数 | [[../algorithms/topological-sort]] |

## P

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| PageRank | — | 网页排名（图算法） | — |
| 路径 | Path | 节点序列 | [[../data-structures/graph]] |
| 路径压缩 | Path Compression | 并查集优化 | [[../data-structures/union-find]] |
| 排列 | Permutation | 全排序 N! 种 | [[../patterns/backtracking]] |
| 前缀和 | Prefix Sum | 累计和 | [[../techniques/prefix-sum]] |
| 优先队列 | Priority Queue | 按优先级 pop | [[../data-structures/heap]] |
| 剪枝 | Pruning | 提前终止无效搜索 | [[../patterns/backtracking]] |

## Q

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| 队列 | Queue | FIFO | [[../data-structures/queue]] |
| 快速选择 | Quickselect | 求第 K 平均 O(N) | [[../algorithms/sorting]] |
| 快速排序 | Quicksort | 分区 + 递归 | [[../algorithms/sorting]] |

## R

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| 基数排序 | Radix Sort | 按位排序 | [[../algorithms/sorting]] |
| 红黑树 | Red-Black Tree | 自平衡 BST，C++ map 用 | [[../data-structures/bst]] |
| 递归 | Recursion | 函数调自己 | [[../concepts/recursion]] |
| 反转 | Reverse | 倒置 | [[../problems/reverse-linked-list]] |
| RoPE | Rotary Position Embedding | （LLM 概念）— |

## S

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| 选择排序 | Selection Sort | 每次找最小 | [[../algorithms/sorting]] |
| 自注意力 | Self-Attention | （LLM 概念）— |
| 线段树 | Segment Tree | 区间最值 / 求和 | [[../data-structures/segment-tree]] |
| 哨兵 | Sentinel | 边界守卫节点 | [[../data-structures/linked-list]] |
| 最短路 | Shortest Path | 从 A 到 B 的最小路径权 | [[../algorithms/shortest-path]] |
| 滑窗 | Sliding Window | 同向双指针 + 窗口状态 | [[../techniques/sliding-window]] |
| 排序 | Sorting | 让数据按某顺序 | [[../algorithms/sorting]] |
| SPFA | Shortest Path Faster Algorithm | Bellman-Ford 队列优化 | [[../algorithms/shortest-path]] |
| 栈 | Stack | LIFO | [[../data-structures/stack]] |
| 状压 | State Compression | 用位掩码当状态 | [[../patterns/dp-bitmask]] |
| 子序列 | Subsequence | 不连续，保序 | [[../problems/lis]] |
| 子串 / 子数组 | Substring / Subarray | 连续 | [[../techniques/sliding-window]] |
| 子集 | Subset | 元素集合 | [[../patterns/backtracking]] |

## T

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| Tarjan | — | 强连通 / 双连通 | — |
| 拓扑排序 | Topological Sort | DAG 上找合法顺序 | [[../algorithms/topological-sort]] |
| Top-K | — | 前 K 大 / 小 | [[../data-structures/heap]] |
| 接雨水 | Trapping Rain Water | 经典单调栈题 | [[../problems/trapping-rain-water]] |
| 树 | Tree | 无环连通图 | [[../data-structures/binary-tree]] |
| Trie | Prefix Tree, 字典树 | 字符为边的树 | [[../data-structures/trie]] |
| 双指针 | Two Pointers | 两个指针协同 | [[../techniques/two-pointers]] |

## U - Z

| 中文 | 英文 | 一句话 | 链接 |
|---|---|---|---|
| 并查集 | Union-Find / DSU | 集合合并 + 查询 | [[../data-structures/union-find]] |
| 顶点 | Vertex | 图的"点" | [[../data-structures/graph]] |
| 通配符 | Wildcard | `?` 或 `*` | — |

---

## 反向链接

- [[../index]] — 主索引
- [[../maps/algorithm-overview]] — 全景图
