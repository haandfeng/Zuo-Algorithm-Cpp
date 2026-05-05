---
title: 图
aliases: [Graph, 邻接表, 邻接矩阵, 有向图, 无向图]
tags: [data-structure, graph]
created: 2026-05-03
updated: 2026-05-03
---

# 图

> **TL;DR**：图 = **节点（顶点）+ 边**。能表达任何"实体 + 关系"，是最通用的数据结构。**怎么存**比"算法本身"更基础——存对了一切都顺。

## 基本术语

- **顶点（vertex）/ 节点（node）**：图中的元素，记 N 个
- **边（edge）**：连接两个顶点，记 M 条
- **有向 / 无向**：边是否带方向
- **带权 / 无权**：边是否有数值
- **度（degree）**：连接到顶点的边数；有向图分入度 / 出度
- **路径（path）**：顶点序列，相邻顶点有边
- **环（cycle）**：起点终点相同的路径
- **连通**（无向）/ **强连通**（有向）：任两点可互达
- **DAG**：有向无环图
- **二分图**：节点能分成两组，边只在组间

## 三种存图方式

### 1. 邻接矩阵 `g[i][j]`

```cpp
vector<vector<int>> g(n, vector<int>(n, 0));
g[u][v] = 1;          // 无权
g[u][v] = w;          // 有权（0 表示无边时要换 INT_MAX）
```

- ✅ 查"u → v 有没有边"O(1)
- ❌ 空间 O(N²)，N=1e5 就爆
- 适合：**N 小（≤ 1000）**、**稠密图**、**Floyd**

### 2. 邻接表 `g[u] = [v1, v2, ...]`

```cpp
vector<vector<int>> g(n);
g[u].push_back(v);              // 无权
g[u].push_back(v); g[v].push_back(u);   // 无向

// 带权
vector<vector<pair<int,int>>> g(n);   // {to, weight}
g[u].push_back({v, w});
```

- ✅ 空间 O(N + M)
- ❌ 查"u → v 有没有边"O(度)
- 适合：**几乎所有场景**

### 3. 链式前向星（竞赛常用）

用 4 个数组 `head/next/to/w` 模拟邻接表，常数最优：

```cpp
int head[N], nxt[M], to[M], w[M], cnt = 0;
void addEdge(int u, int v, int weight) {
    to[++cnt] = v; w[cnt] = weight;
    nxt[cnt] = head[u]; head[u] = cnt;
}
// 遍历 u 的出边
for (int e = head[u]; e; e = nxt[e]) {
    int v = to[e], weight = w[e];
}
```

竞赛/算法题首选，工程少见。

## 图的两种基本遍历

### BFS（用队列）

```cpp
void bfs(int s, vector<vector<int>>& g) {
    int n = g.size();
    vector<bool> vis(n, false);
    queue<int> q;
    q.push(s); vis[s] = true;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        // 处理 u
        for (int v : g[u]) {
            if (!vis[v]) { vis[v] = true; q.push(v); }
        }
    }
}
```

特性：**逐层扩展**——天然能算"无权图最短路（边数最少）"。

### DFS（用栈或递归）

```cpp
vector<bool> vis;
void dfs(int u, vector<vector<int>>& g) {
    vis[u] = true;
    // 处理 u
    for (int v : g[u]) if (!vis[v]) dfs(v, g);
}
```

特性：**深入到底再回退**——适合**连通块、拓扑、回溯**。

详见 [[bfs-dfs]]。

## 图的几个核心问题

| 问题 | 推荐算法 |
|---|---|
| 连通块个数 | DFS / BFS / [[union-find\|并查集]] |
| 无权最短路 | BFS |
| 带权最短路（非负权） | [[shortest-path\|Dijkstra]] |
| 带权最短路（有负权） | [[shortest-path\|Bellman-Ford / SPFA]] |
| 多源最短路 | [[shortest-path\|Floyd]] |
| 最小生成树 | [[mst\|Kruskal / Prim]] |
| 拓扑排序 | [[topological-sort\|Kahn / DFS]] |
| 强连通分量 | Tarjan / Kosaraju |
| 二分图判定 | BFS / DFS 染色 |
| 二分图匹配 | 匈牙利 / Hopcroft-Karp |
| 网络流 | Dinic / EK |

## 何时想到图

| 信号 | 抽象 |
|---|---|
| "节点 + 边 + 关系" | 直接是图 |
| "课程依赖 / 编译顺序" | DAG + 拓扑 |
| "网格上下左右移动" | 隐式图 |
| "状态空间搜索" | BFS：状态是节点，转移是边 |
| "字典 / 单词变换" | BFS（[[problems/word-ladder\|单词接龙]]） |
| "区域连通" | DFS / 并查集 |

## 网格图技巧

二维网格本身就是图，常用方向数组：

```cpp
int dx[] = {-1, 1, 0, 0};
int dy[] = {0, 0, -1, 1};

for (int d = 0; d < 4; ++d) {
    int nx = x + dx[d], ny = y + dy[d];
    if (nx < 0 || nx >= n || ny < 0 || ny >= m) continue;
    if (visited[nx][ny] || grid[nx][ny] == 0) continue;
    // 递归 / 入队
}
```

经典题：岛屿数量 LC 200、被围绕的区域 LC 130、矩阵中的最长递增路径 LC 329。

## 入度 / 出度（有向图）

```cpp
vector<int> in(n), out(n);
for (int u = 0; u < n; ++u) {
    out[u] = g[u].size();
    for (int v : g[u]) in[v]++;
}
```

`in[v] == 0` 的节点是**源点**——拓扑排序的起点。

## 易错点

- 无向图加边要**两次** `g[u].push_back(v); g[v].push_back(u)`
- 矩阵存图时**对角线**是不是 0 还是 INF 要看题
- BFS / DFS 忘加 `visited` → 死循环或重复
- 节点编号从 0 还是 1 开始 → 数组大小差 1
- 自环 / 重边要不要去掉 → 看题
- 网格越界检查放在**入队前**而不是**出队后**——能减少很多 vis 状态

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「图」相关关键词（图）的笔记，按来源分组。

### 灵茶山艾府和代码随想录
- [[图|课程/灵茶山艾府和代码随想录/图.md]]

### 左程云
- [[_图论知识点|课程/左程云/001. 难点重点照顾/_图论知识点.md]]
- [[Java中位图的实现|课程/左程云/002. 知识点总结/00. 编程语言相关/Java中位图的实现.md]]
- [[位图|课程/左程云/002. 知识点总结/20. 算法知识点/位图.md]]
- [[图&树|课程/左程云/002. 知识点总结/20. 算法知识点/图&树.md]]
- [[图的拓扑排序算法|课程/左程云/010. 代码模板/图的拓扑排序算法.md]]
- [[图的统一表述结构|课程/左程云/010. 代码模板/图的统一表述结构.md]]

### labuladong
- [[DFS 图回溯 岛屿 相关 问题|课程/labuladong/经典暴力搜索算法/DFS 图回溯 岛屿 相关 问题.md]]
- [[图拓扑排序 和 并查集|课程/labuladong/经典暴力搜索算法/图拓扑排序 和 并查集.md]]

### 题目
- [[0048-旋转图像|题目/0048-旋转图像.md]]
- [[0084-柱状图中最大的矩形|题目/0084-柱状图中最大的矩形.md]]
- [[0133-克隆图|题目/0133-克隆图.md]]
- [[0199-二叉树的右视图|题目/0199-二叉树的右视图.md]]
- [[0733-图像渲染|题目/0733-图像渲染.md]]
- [[1971-寻找图中是否存在路径|题目/1971-寻找图中是否存在路径.md]]

## 参考资料

- [[../../课程/灵茶山艾府和代码随想录/图]] — 你整理的图专题
- [[../../课程/左程云/100. 算法课程]] — 左程云图论章节
- 《算法竞赛进阶指南》第 3 章

## 关联条目

- 上层：[[ds-overview]]
- 算法：[[bfs-dfs]] · [[topological-sort]] · [[shortest-path]] · [[mst]]
- 数据结构：[[union-find]] · [[queue]] · [[stack]]
