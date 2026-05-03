---
title: 最小生成树
aliases: [MST, Minimum Spanning Tree, Kruskal, Prim]
tags: [algorithm, graph]
created: 2026-05-03
updated: 2026-05-03
---

# 最小生成树（MST）

> **TL;DR**：给定带权无向连通图，找出**包含所有顶点、边权总和最小**的树（N 个点 N-1 条边）。两个经典算法：**Kruskal**（加边 + [[../data-structures/union-find|并查集]]）和 **Prim**（加点 + [[../data-structures/heap|堆]]）。

## Kruskal

**思想**：把所有边按权重排序，从小到大枚举，**只要不形成环就加入** MST。用并查集 O(α(N)) 判环。

```cpp
struct Edge { int u, v, w; };
class DSU { /* see [[../data-structures/union-find]] */ };

int kruskal(int n, vector<Edge>& edges) {
    sort(edges.begin(), edges.end(),
         [](const Edge& a, const Edge& b) { return a.w < b.w; });
    DSU dsu(n);
    int total = 0, picked = 0;
    for (auto& e : edges) {
        if (dsu.unite(e.u, e.v)) {
            total += e.w;
            picked++;
            if (picked == n - 1) break;
        }
    }
    return picked == n - 1 ? total : -1;     // -1 表示不连通
}
```

复杂度 O(M log M)。**适合稀疏图**。

## Prim

**思想**：从任一点出发，**每次把"距已建树最近"的点加入**，类似 [[shortest-path|Dijkstra]]。

```cpp
int prim(int n, vector<vector<pair<int,int>>>& g) {
    vector<int> dist(n, INT_MAX);
    vector<bool> inMST(n, false);
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    dist[0] = 0;
    pq.push({0, 0});
    int total = 0, picked = 0;
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (inMST[u]) continue;
        inMST[u] = true;
        total += d;
        picked++;
        for (auto [v, w] : g[u]) {
            if (!inMST[v] && w < dist[v]) {
                dist[v] = w;
                pq.push({w, v});
            }
        }
    }
    return picked == n ? total : -1;
}
```

复杂度 O((N + M) log N)。**适合稠密图**——朴素 O(N²) 版本对 N ≤ 5000 也很快。

## Kruskal vs Prim

| 维度 | Kruskal | Prim |
|---|---|---|
| 思想 | 加边 + 不成环 | 加点 + 最近邻 |
| 数据结构 | 排序 + 并查集 | 优先队列 |
| 复杂度 | O(M log M) | O((N + M) log N) |
| 适合 | 稀疏图 | 稠密图 |
| 实现 | 易（5 行） | 中等 |

## 经典题型

| 题 | 备注 |
|---|---|
| 修建公路 | 模板 |
| 连接所有点的最小费用 LC 1584 | 完全图 → Prim |
| 连通图最小代价 | Kruskal |
| 最小生成树（HDU） | 模板 |
| 次小生成树 | 先 MST，再枚举非树边 + 树上最大边 |
| 瓶颈 MST | MST 的最大边 = 任意两点路径上最大边的最小值 |

## 关键性质

- **环切原则（cut property）**：任一环上权重**最大的边一定不在 MST**
- **割切原则**：任一割（vertex partition）的**最小横跨边一定在 MST**
- N 个点的 MST 恰好 N-1 条边
- MST **不一定唯一**，但**所有 MST 的边权多重集相同**

## 易错点

- 输入是**有向图**？MST 通常是无向；有向是"最小树形图"（朱刘算法）
- 不连通 → 返回 -1 或 INF
- 边数 ≠ N - 1 → 没生成树
- Prim 朴素版用 `dist[]`，堆版要"过期检查"避免重复

## 参考资料

- [[../../算法笔记/zuoAlgorithm/100. 算法课程]] — 左程云 MST 章节
- 《算法竞赛进阶指南》MST 章节

## 关联条目

- 上层：[[../maps/algorithm-overview]]
- 数据结构：[[../data-structures/graph]] · [[../data-structures/union-find]] · [[../data-structures/heap]]
- 兄弟：[[shortest-path]]
- 思维：[[../patterns/greedy]]
