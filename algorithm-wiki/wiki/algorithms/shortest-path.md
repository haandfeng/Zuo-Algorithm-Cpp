---
title: 最短路
aliases: [Shortest Path, Dijkstra, Bellman-Ford, SPFA, Floyd]
tags: [algorithm, graph]
created: 2026-05-03
updated: 2026-05-03
---

# 最短路

> **TL;DR**：四个核心算法对应四种场景：**单源非负权 → Dijkstra**；**单源含负权 → Bellman-Ford / SPFA**；**多源 → Floyd**；**无权 → BFS**。

## 速查表

| 算法 | 复杂度 | 适用场景 | 能处理负权 | 能检测负环 |
|---|---|---|---|---|
| BFS | O(N + M) | 无权 / 边权全 1 | — | — |
| Dijkstra（堆优化） | O((N + M) log N) | 单源 + **非负**权 | ❌ | ❌ |
| Bellman-Ford | O(NM) | 单源 + **可负**权 | ✅ | ✅ |
| SPFA | 平均 O(kM)，最坏 O(NM) | Bellman-Ford 优化 | ✅ | ✅ |
| Floyd | O(N³) | **多源**（任意点对） | ✅ | ✅（对角线 < 0） |

## Dijkstra（最常用）

**贪心 + 优先队列**：每次取出当前距离最小的未确定节点，确定它的最短距离，然后松弛它的邻居。

```cpp
vector<int> dijkstra(int s, int n, vector<vector<pair<int,int>>>& g) {
    // g[u] = {{v, w}, ...}
    vector<int> dist(n, INT_MAX);
    dist[s] = 0;
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    pq.push({0, s});       // {距离, 节点}
    while (!pq.empty()) {
        auto [d, u] = pq.top(); pq.pop();
        if (d > dist[u]) continue;       // 过期记录
        for (auto [v, w] : g[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                pq.push({dist[v], v});
            }
        }
    }
    return dist;
}
```

要点：
- **必须非负权**——负权会破坏贪心正确性
- 用堆优化：每次 O(log N) 找最小，总 O((N + M) log N)
- "过期记录"用 `d > dist[u]` 跳过——比 vis 数组更省事

### 输出路径

记录前驱：

```cpp
vector<int> prev(n, -1);
// 在 dist[v] = dist[u] + w 时记录 prev[v] = u

vector<int> path(int t) {
    vector<int> p;
    for (int u = t; u != -1; u = prev[u]) p.push_back(u);
    reverse(p.begin(), p.end());
    return p;
}
```

## Bellman-Ford

**所有边松弛 N-1 次**，纯暴力但能处理负权和负环：

```cpp
vector<int> bellmanFord(int s, int n, vector<tuple<int,int,int>>& edges) {
    vector<int> dist(n, INT_MAX);
    dist[s] = 0;
    for (int i = 0; i < n - 1; ++i) {
        for (auto [u, v, w] : edges) {
            if (dist[u] != INT_MAX && dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
            }
        }
    }
    // 第 N 次还能松弛 → 有负环
    for (auto [u, v, w] : edges) {
        if (dist[u] != INT_MAX && dist[u] + w < dist[v]) return {};   // 负环
    }
    return dist;
}
```

时间 O(NM)，简单粗暴。常用于：

- 限制经过最多 K 条边的最短路（LC 787）
- 判负环
- 边数少（M < 1e4）

## SPFA

Bellman-Ford 的队列优化：**只有距离被更新的节点才需要再松弛**：

```cpp
vector<int> spfa(int s, int n, vector<vector<pair<int,int>>>& g) {
    vector<int> dist(n, INT_MAX);
    vector<bool> in_queue(n, false);
    dist[s] = 0;
    queue<int> q;
    q.push(s); in_queue[s] = true;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        in_queue[u] = false;
        for (auto [v, w] : g[u]) {
            if (dist[u] + w < dist[v]) {
                dist[v] = dist[u] + w;
                if (!in_queue[v]) { q.push(v); in_queue[v] = true; }
            }
        }
    }
    return dist;
}
```

平均很快，但最坏退化到 O(NM)。**算法竞赛已不推荐**（容易被卡），工程也少用。

## Floyd

三重循环求**所有点对最短路**：

```cpp
vector<vector<int>> floyd(int n, vector<vector<int>>& g) {
    vector<vector<int>> d = g;       // d[i][j] 初始是直连边
    for (int k = 0; k < n; ++k)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (d[i][k] != INT_MAX && d[k][j] != INT_MAX)
                    d[i][j] = min(d[i][j], d[i][k] + d[k][j]);
    return d;
}
```

时间 O(N³)，空间 O(N²)。

要点：
- **k 必须在最外层**——表示"允许中间经过 0..k 的节点"
- 适合 N ≤ 500
- 也能判负环：`d[i][i] < 0`
- 是经典 [[dp|DP]]：状态 = "i 到 j 经过 ≤ k 的最短距离"

## 0-1 BFS

边权只有 0 和 1 时，用 **deque** 替代普通 queue：

- 边权 0 → push_front
- 边权 1 → push_back

复杂度 O(N + M)，比 Dijkstra 省 log。

## 选型决策

```text
有边权？
├─ 否 → BFS
└─ 是 → 全部非负？
    ├─ 是 → 单源还是多源？
    │   ├─ 单源 → Dijkstra
    │   └─ 多源 + N ≤ 500 → Floyd
    └─ 有负权？
        ├─ 限制边数 → Bellman-Ford
        ├─ 单源 + 稠密 → Bellman-Ford
        └─ 单源 + 稀疏 + 不会被卡 → SPFA
```

## 经典题型

| 题 | 用法 |
|---|---|
| 网络延迟时间 LC 743 | Dijkstra |
| 概率最大的路径 LC 1514 | Dijkstra（边权改成乘） |
| K 站中转内最便宜的航班 LC 787 | Bellman-Ford（限边数） |
| 阈值距离内邻居最少的城市 LC 1334 | Floyd |
| 修建公路（最小生成树） | 不是最短路，是 [[mst]] |
| 二叉树最近公共祖先 LC 236 | 不是最短路，是树上 LCA |
| 找到小镇的法官 LC 997 | 不是最短路，是入度出度 |

## 易错点

- Dijkstra **用了负权 → 错**
- 优先队列堆顶比较方向反了 → 大顶堆 / 小顶堆搞反
- INT_MAX 加东西**溢出** → 用 `if (dist[u] != INT_MAX) check` 或者用 `long long`
- 同一节点**多次入堆**没事，但要用过期检查跳过
- Floyd 的 `k` 必须**最外层**——内外搞反结果错
- 链式前向星初始化 `head[]` 要清零或 -1（看哨兵约定）

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「最短路」相关关键词（最短路, Dijkstra, Floyd）的笔记，按来源分组。

### 左程云
- [[Dijkstra算法|课程/左程云/002. 知识点总结/20. 算法知识点/Dijkstra算法.md]]
- [[单源最短路径算法|课程/左程云/002. 知识点总结/20. 算法知识点/单源最短路径算法.md]]
- [[Dijkstra算法代码|课程/左程云/010. 代码模板/Dijkstra算法代码.md]]
- [[施展一次魔法的最短路|课程/左程云/200. ★算法题目汇总★/★算法面试题汇总/施展一次魔法的最短路.md]]

## 参考资料

- [[../../课程/左程云/100. 算法课程]] — 左程云图论章节
- [[../../课程/灵茶山艾府 + 代码随想录/图]] — 你的图专题
- 《算法竞赛进阶指南》最短路章节

## 关联条目

- 上层：[[algorithm-overview]]
- 数据结构：[[graph]] · [[heap]]
- 兄弟：[[mst]] · [[bfs-dfs]] · [[topological-sort]]
