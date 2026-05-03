---
title: 拓扑排序
aliases: [Topological Sort, Topo Sort, Kahn]
tags: [algorithm, graph]
created: 2026-05-03
updated: 2026-05-03
---

# 拓扑排序

> **TL;DR**：在 **DAG（有向无环图）** 上，求一个节点序列使得**每条边都从前指向后**。两种经典实现：**Kahn（BFS）** 和 **DFS 后序逆序**。常用于"依赖 / 调度 / 编译顺序"问题。

## 直觉

如果问题中存在 "A 必须在 B 之前" 这种依赖，那就是 DAG，拓扑排序给出一个合法执行顺序。

只要有**环**就**没有**拓扑序——拓扑排序常被用来**判环**。

## Kahn 算法（BFS 版，最常用）

核心思想：**所有入度为 0 的节点先入队**，每次出队就把它的出边删掉，更新邻居入度，新的 0 入度入队。

```cpp
vector<int> topoSort(int n, vector<vector<int>>& g) {
    vector<int> in(n, 0);
    for (int u = 0; u < n; ++u)
        for (int v : g[u]) in[v]++;

    queue<int> q;
    for (int i = 0; i < n; ++i) if (in[i] == 0) q.push(i);

    vector<int> order;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        order.push_back(u);
        for (int v : g[u]) {
            if (--in[v] == 0) q.push(v);
        }
    }

    if ((int)order.size() != n) return {};   // 有环！返回空表示无解
    return order;
}
```

时间 O(N + M)，空间 O(N)。

## DFS 版（后序逆序）

```cpp
bool dfs(int u, vector<vector<int>>& g, vector<int>& vis, vector<int>& order) {
    vis[u] = 1;                  // 1 = 正在访问
    for (int v : g[u]) {
        if (vis[v] == 1) return false;       // 撞到了正在访问的 → 有环
        if (vis[v] == 0 && !dfs(v, g, vis, order)) return false;
    }
    vis[u] = 2;                  // 2 = 已完成
    order.push_back(u);          // 后序加入
    return true;
}

vector<int> topoSort(int n, vector<vector<int>>& g) {
    vector<int> vis(n, 0), order;
    for (int i = 0; i < n; ++i) {
        if (vis[i] == 0 && !dfs(i, g, vis, order)) return {};
    }
    reverse(order.begin(), order.end());     // 后序逆序就是拓扑序
    return order;
}
```

DFS 版的"3 色"标记法：白（未访问）→ 灰（正在访问）→ 黑（已完成）。**遇灰即环**。

## 何时想到拓扑排序

| 信号 | 应用 |
|---|---|
| "X 必须在 Y 之前" | 任务 / 课程依赖 |
| "判断 DAG 中有没有环" | Kahn 或 DFS 三色 |
| "DAG 上的 DP" | DAG-DP，按拓扑序遍历 |
| "字典序最小拓扑序" | Kahn + 优先队列（小顶堆） |
| "构造唯一序" | Kahn 中入队个数始终 ≤ 1 |

## 经典题型

| 题 | 用法 |
|---|---|
| 课程表 LC 207 | 判环 |
| 课程表 II LC 210 | 输出拓扑序 |
| 火星词典 LC 269 | 字符比较推依赖 → 拓扑 |
| 课程表 IV LC 1462 | 拓扑 + 闭包 |
| 找到最终的安全状态 LC 802 | 反向拓扑 |
| 序列重建 LC 444 | 验证拓扑序唯一性 |
| 最小高度树 LC 310 | 不断剥叶子（类拓扑） |
| 任务调度 / 项目管理 | DAG-DP，按拓扑序算最早 / 最晚开始 |

### 字典序最小拓扑序

把 BFS 的 `queue` 换成 `priority_queue`（小顶堆）：

```cpp
priority_queue<int, vector<int>, greater<int>> pq;
for (int i = 0; i < n; ++i) if (in[i] == 0) pq.push(i);

vector<int> order;
while (!pq.empty()) {
    int u = pq.top(); pq.pop();
    order.push_back(u);
    for (int v : g[u]) if (--in[v] == 0) pq.push(v);
}
```

时间 O((N + M) log N)。

## 拓扑序唯一的判定

拓扑序唯一 ⟺ Kahn 过程中**每一步队列里至多 1 个节点**。多于 1 个就说明有"自由顺序"。

## 与其它算法的关系

- 是 [[bfs-dfs|BFS/DFS]] 的特化
- 跟 [[union-find|并查集]] **不一样**：并查集处理无向图连通性，拓扑处理有向图顺序
- DAG 上 [[shortest-path|最短路]] 可以**按拓扑序 DP**，O(N + M)，比 Dijkstra 还快

## 易错点

- 输入是有向图——加边只加一次 `g[u].push_back(v)`，**不要**双向加
- Kahn 的环检测靠"入队个数 ≠ N"，不是看 in 数组
- DFS 三色法中 `vis[u] = 1` 和 `vis[u] = 2` **不能合并**——合并就检测不到环
- 节点编号 0 / 1 起算要统一
- 并行任务调度题如果允许"批次"，结果不是单一拓扑序而是**层数**

## 参考资料

- [[../../课程/灵茶山艾府 + 代码随想录/图]] — 你的图专题
- [[../../课程/左程云/100. 算法课程]] — 左程云拓扑章节

## 关联条目

- 上层：[[algorithm-overview]]
- 数据结构：[[graph]] · [[queue]]
- 兄弟：[[bfs-dfs]] · [[shortest-path]]
- 应用：[[dp]]（DAG-DP）
