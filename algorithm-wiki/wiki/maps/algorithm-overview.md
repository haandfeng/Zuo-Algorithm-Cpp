---
title: 算法学习全景图
tags: [moc, map, overview]
updated: 2026-05-03
---

# 算法学习全景图

> 一张 mermaid 把整个 Wiki 的核心概念串起来。
> 每个节点对应一个 `wiki/` 下的具体条目。

```mermaid
flowchart TB
    subgraph 基础
        Complexity[复杂度]
        Recursion[递归]
        DC[分治]
    end

    subgraph 数据结构
        Array[数组]
        LL[链表]
        Stack[栈]
        Queue[队列]
        Hash[哈希表]
        Tree[二叉树]
        Heap[堆]
        UF[并查集]
        Graph[图]
        Trie[Trie]
    end

    subgraph 算法
        Sort[排序]
        BS[二分查找]
        BFSDFS[BFS/DFS]
        Topo[拓扑排序]
        SP[最短路]
        MST[最小生成树]
        KMP[KMP]
        Bit[位运算]
    end

    subgraph 技巧
        TP[双指针]
        SW[滑动窗口]
        PS[前缀和]
        MS[单调栈]
        MQ[单调队列]
        BSA[二分答案]
    end

    subgraph 范式
        Greedy[贪心]
        BT[回溯]
        DP[动态规划]
    end

    Complexity --> Sort
    Recursion --> DC --> Sort
    Recursion --> Tree --> BFSDFS
    Recursion --> BT
    Recursion --> DP

    Array --> TP --> SW
    Array --> PS
    Array --> Sort

    Stack --> MS
    Queue --> MQ --> SW

    Tree --> BFSDFS
    Graph --> BFSDFS --> Topo
    Graph --> SP --> MST

    Hash --> SW
    Hash --> Trie

    BS --> BSA
    Sort --> BS
```

## 阅读建议

- **零基础**：先 `基础` → `数据结构` 主干（数组 / 链表 / 栈队列 / 哈希 / 树）→ `算法` 主干（排序 / 二分 / DFS / BFS）
- **冲刺面试**：补 `技巧` 全部 + `范式` 中的回溯 + DP 入门 → 然后做 [[../../题单/Hot100|Hot100]]
- **进阶**：`算法` 的 [[shortest-path]] / [[kmp]] / [[manacher]] + `范式` 的 [[dp-knapsack]] / [[dp-tree]] / [[dp-bitmask]]

## 与刷题路线的关系

本图按"概念依赖"组织。如果你想要"按题号刷"，请使用：

- [[../../题单/Hot100|LeetCode Hot 100]]
- [[../../题单/Blind75|Blind 75]]
- [[../../题单/Grind75|Grind 75]]
- [[../../题单/Leetcode150|LC Top Interview 150]]

或者按**专题**刷：
- [[../../课程/灵茶山艾府 + 代码随想录/动态规划|代码随想录 DP]]
- [[../../课程/灵茶山艾府 + 代码随想录/二叉树|代码随想录 二叉树]]

本 Wiki 提供"概念入口"，刷题笔记在 Vault 别的位置。

## 相关地图

- [[ds-overview]] — 只看数据结构
- [[dp-roadmap]] — 只看 DP
