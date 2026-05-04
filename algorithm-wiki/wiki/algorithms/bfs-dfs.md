---
title: BFS / DFS
aliases: [BFS, DFS, 广度优先, 深度优先, 图遍历]
tags: [algorithm, search, graph]
created: 2026-05-03
updated: 2026-05-03
---

# BFS / DFS

> **TL;DR**：BFS（广度优先）和 DFS（深度优先）是图与树的两种基本遍历。**BFS 用队列、逐层扩展、自带最短路（无权图）**；**DFS 用栈/递归、深入到底、适合连通性 / 拓扑 / 回溯**。

## BFS 模板

```cpp
void bfs(int start, vector<vector<int>>& g) {
    int n = g.size();
    vector<int> dist(n, -1);
    queue<int> q;
    q.push(start);
    dist[start] = 0;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        for (int v : g[u]) {
            if (dist[v] != -1) continue;          // 已访问
            dist[v] = dist[u] + 1;
            q.push(v);
        }
    }
}
```

**关键不变量**：BFS 第一次访问到 v 时，`dist[v]` 就是 start 到 v 的**最少边数**。这是 BFS 自带最短路的来源。

### 分层 BFS

如果要"按层处理"（如二叉树层序、求最大宽度）：

```cpp
while (!q.empty()) {
    int sz = q.size();              // 固定本层大小
    for (int i = 0; i < sz; ++i) {
        auto u = q.front(); q.pop();
        // 处理 u
        for (auto v : neighbors(u)) {
            if (!vis[v]) { vis[v] = true; q.push(v); }
        }
    }
    // 一层处理完
    level++;
}
```

### 双向 BFS

如果起点和终点都已知，**从两端同时扩**，相遇就停。把搜索空间从 $b^d$ 降到 $2 b^{d/2}$，对大状态空间是数量级提升。

详见 [[problems/word-ladder]]。

## DFS 模板

### 递归版

```cpp
vector<bool> vis;
void dfs(int u, vector<vector<int>>& g) {
    vis[u] = true;
    // 处理 u
    for (int v : g[u]) {
        if (!vis[v]) dfs(v, g);
    }
}
```

### 迭代版（显式栈）

```cpp
void dfs(int start, vector<vector<int>>& g) {
    stack<int> st;
    st.push(start);
    vector<bool> vis(g.size(), false);
    while (!st.empty()) {
        int u = st.top(); st.pop();
        if (vis[u]) continue;
        vis[u] = true;
        for (int v : g[u]) if (!vis[v]) st.push(v);
    }
}
```

⚠️ 迭代版的访问顺序和递归版**不完全相同**——题目要求确定访问顺序时要小心。

## BFS vs DFS 选型

| 维度 | BFS | DFS |
|---|---|---|
| 数据结构 | 队列 | 栈 / 递归 |
| 空间 | O(W)，W 是层最大宽度 | O(D)，D 是最大深度 |
| 适合 | 最短路、最少步数、按层处理 | 连通块、回溯、拓扑、欧拉路径 |
| 找解 | **最短路**自然出 | 找**任意一条**路径快 |
| 树高 / 宽很大时 | DFS 内存少（深 ≤ 宽时） | BFS 内存少（宽 ≤ 深时） |

实战经验：

- **求最少步数 / 最短距离 / 最短路径长度** → **BFS**
- **求所有可能路径 / 子集 / 排列 / 组合** → **DFS（[[backtracking]]）**
- **求连通块 / 染色** → 任选一个，DFS 写起来短
- **拓扑排序** → BFS（Kahn）或 DFS

## 网格问题模板

二维网格当作隐式图：

```cpp
int dx[4] = {-1, 1, 0, 0}, dy[4] = {0, 0, -1, 1};

void dfs(int x, int y, vector<vector<int>>& g) {
    int n = g.size(), m = g[0].size();
    if (x < 0 || x >= n || y < 0 || y >= m) return;
    if (g[x][y] != 1) return;
    g[x][y] = 2;                    // 标记访问（或用 visited 数组）
    for (int d = 0; d < 4; ++d) dfs(x + dx[d], y + dy[d], g);
}
```

**陷阱**：8 方向问题时方向数组扩到 8 个；对角线相邻的题（如某些岛屿题）容易漏。

## 经典题型

| 题 | BFS / DFS | 备注 |
|---|---|---|
| 二叉树层序遍历 LC 102 | BFS | 分层 |
| 二叉树最大深度 LC 104 | DFS | 后序 |
| 岛屿数量 LC 200 | DFS / BFS / [[union-find\|并查集]] | 连通块 |
| 岛屿最大面积 LC 695 | DFS | 累计 |
| 被围绕的区域 LC 130 | DFS | 反向：从边界开始标记 |
| 单词接龙 LC 127 | BFS / 双向 BFS | 见 [[problems/word-ladder]] |
| 01 矩阵 LC 542 | 多源 BFS | 把所有 0 同时入队 |
| 课程表 LC 207 | 拓扑（BFS / DFS） | 见 [[topological-sort]] |
| 克隆图 LC 133 | DFS / BFS + hash | 复制节点 |
| 太平洋大西洋水流 LC 417 | 双向 DFS | 反向思考 |
| 子集 LC 78 | DFS / [[backtracking]] | 不是图，但用 DFS |

## 多源 BFS

把**多个起点**同时入队、各自的 dist=0：

```cpp
queue<pair<int,int>> q;
for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
        if (g[i][j] == 0) {           // 多个 0 都是源
            dist[i][j] = 0;
            q.push({i, j});
        }
while (!q.empty()) {
    auto [x, y] = q.front(); q.pop();
    for (int d = 0; d < 4; ++d) {
        int nx = x + dx[d], ny = y + dy[d];
        if (in_bounds && dist[nx][ny] == -1) {
            dist[nx][ny] = dist[x][y] + 1;
            q.push({nx, ny});
        }
    }
}
```

LC 542、LC 994 都是经典多源 BFS。

## 易错点

- BFS 忘了**入队时**标记 visited → 同一节点多次入队，TLE
- DFS 递归深度过深 → 爆栈（C++ 默认栈 1MB，深度 ~1e5 危险）
- 网格题 visited 用 `g[x][y] = 2` 改原数组 → **修改了原数据**，注意题目允不允许
- 双向 BFS 两边的访问标记要分开
- 0-1 BFS（边权 0 / 1）应该用 deque，不是普通 queue

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「BFS/DFS」相关关键词（BFS, DFS, 广度, 深度, 搜索）的笔记，按来源分组。

### 左程云
- [[拓扑排序DFS解法|课程/左程云/002. 知识点总结/20. 算法知识点/拓扑排序DFS解法.md]]
- [[搜索二叉树|课程/左程云/002. 知识点总结/20. 算法知识点/搜索二叉树.md]]
- [[深度优先遍历|课程/左程云/002. 知识点总结/20. 算法知识点/深度优先遍历.md]]
- [[宽度优先遍历深度优先遍历|课程/左程云/004. 刷题经验总结/宽度优先遍历深度优先遍历.md]]
- [[记忆化搜索跟经典动态规划|课程/左程云/004. 刷题经验总结/记忆化搜索跟经典动态规划.md]]
- [[不同的二叉搜索树|课程/左程云/200. ★算法题目汇总★/★算法面试题汇总/不同的二叉搜索树.md]]

### labuladong
- [[BFS|课程/labuladong/经典暴力搜索算法/BFS.md]]
- [[DFS 图回溯 岛屿 相关 问题|课程/labuladong/经典暴力搜索算法/DFS 图回溯 岛屿 相关 问题.md]]

### 题目
- [[0033-搜索旋转排序数组|题目/0033-搜索旋转排序数组.md]]
- [[0096-不同的二叉搜索树|题目/0096-不同的二叉搜索树.md]]
- [[0098-验证二叉搜索树|题目/0098-验证二叉搜索树.md]]
- [[0104-二叉树的最大深度|题目/0104-二叉树的最大深度.md]]
- [[0108-将有序数组转换为二叉搜索树|题目/0108-将有序数组转换为二叉搜索树.md]]
- [[0111-二叉树的最小深度|题目/0111-二叉树的最小深度.md]]

## 参考资料

- [[../../课程/灵茶山艾府和代码随想录/图]] — 你的图遍历专题
- [[../../课程/labuladong/经典暴力搜索算法]] — labuladong DFS / BFS
- [[../../课程/左程云/100. 算法课程]] — 左程云图论

## 关联条目

- 上层：[[algorithm-overview]]
- 数据结构：[[graph]] · [[queue]] · [[stack]] · [[binary-tree]]
- 衍生：[[topological-sort]] · [[shortest-path]] · [[backtracking]]
- 应用：[[problems/word-ladder]]
