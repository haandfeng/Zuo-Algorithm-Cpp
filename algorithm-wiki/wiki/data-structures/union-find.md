---
title: 并查集
aliases: [Union-Find, DSU, Disjoint Set Union, 并查集]
tags: [data-structure, set]
created: 2026-05-03
updated: 2026-05-03
---

# 并查集（Union-Find / DSU）

> **TL;DR**：并查集维护**若干互不相交的集合**，支持两个核心操作：**find(x)** 找 x 所在集合的代表元、**union(x, y)** 合并两个集合。配合**路径压缩 + 按秩合并**后单次操作均摊近 O(1)。

## 用一棵"森林"表达多个集合

每个集合是一棵树，树根作为代表元：

```text
集合 1：       3        集合 2：    5
              / \                    \
             1   4                    7
            /
           2
代表元 = 3              代表元 = 5
```

数组 `parent[i]` 存 i 的父节点，根节点的 `parent[i] = i`。

## 朴素实现

```cpp
class DSU {
    vector<int> parent;
public:
    DSU(int n) : parent(n) { iota(parent.begin(), parent.end(), 0); }   // parent[i] = i

    int find(int x) {
        while (parent[x] != x) x = parent[x];
        return x;
    }

    void unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px != py) parent[px] = py;
    }
};
```

朴素版最坏 O(N) — 容易变成长链。

## 两个优化

### 1. 路径压缩（Path Compression）

`find` 时把路径上所有节点直接挂到根：

```cpp
int find(int x) {
    return parent[x] == x ? x : parent[x] = find(parent[x]);   // 递归 + 顺手改
}
```

### 2. 按秩 / 按大小合并（Union by Rank / Size）

合并时把**矮的挂到高的**下面，避免增加深度：

```cpp
class DSU {
    vector<int> parent, sz;
public:
    DSU(int n) : parent(n), sz(n, 1) { iota(parent.begin(), parent.end(), 0); }

    int find(int x) {
        return parent[x] == x ? x : parent[x] = find(parent[x]);
    }

    bool unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;
        if (sz[px] < sz[py]) swap(px, py);    // 大的当根
        parent[py] = px;
        sz[px] += sz[py];
        return true;
    }

    bool connected(int x, int y) { return find(x) == find(y); }
    int component_size(int x) { return sz[find(x)]; }
};
```

**两优化合用**后单次操作均摊 O(α(N))，α 是反阿克曼函数，实际上 ≤ 4。

## 复杂度

| 操作 | 朴素 | 路径压缩 | 路径压缩 + 按秩 |
|---|---|---|---|
| find | O(N) | 均摊 O(log N) | 均摊 O(α(N)) ≈ O(1) |
| unite | O(N) | 均摊 O(log N) | 均摊 O(α(N)) ≈ O(1) |

## 何时想到并查集

| 信号 | 例子 |
|---|---|
| "判断两点是否连通" | 朋友圈、岛屿数量、网络连接 |
| "动态合并集合" | Kruskal 最小生成树 |
| "等价类划分" | 字符串等价、变量同名 |
| "离线连通性查询" | 把删边离线成加边 |
| "环检测（无向图）" | unite 时已连通就有环 |

**反例**：若需要"分裂"集合（split / 撤销），并查集**不擅长**——要用**可撤销并查集**或别的数据结构。

## 经典题型

| 题 | 用法 |
|---|---|
| 朋友圈 LC 547 | unite 所有有关系的对，数根的个数 |
| 岛屿数量 LC 200 | 也可以 [[bfs-dfs\|DFS/BFS]]；并查集是另一解 |
| 冗余连接 LC 684 | 加边时已连通就是答案 |
| 账户合并 LC 721 | 邮箱 → 账户根，再聚合 |
| 等式方程的可满足性 LC 990 | 等号合并、不等号检查 |
| 最长连续序列 LC 128 | 哈希 + 并查集 |
| Kruskal 最小生成树 | 排序边 + 并查集 |
| 网络延迟 LC 743 | （这题用 Dijkstra，提示并查集不适合最短路） |
| 由斜杠划分区域 LC 959 | 把每格分 4 块用并查集合并 |

### 模板：岛屿数量并查集解

```cpp
int numIslands(vector<vector<char>>& g) {
    int n = g.size(), m = g[0].size();
    DSU dsu(n * m + 1);
    int water = n * m;            // 虚拟"海"节点
    auto id = [&](int i, int j) { return i * m + j; };
    int land = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            if (g[i][j] == '0') continue;
            land++;
            for (auto [di, dj] : vector<pair<int,int>>{{-1,0},{0,-1}}) {
                int ni = i + di, nj = j + dj;
                if (ni >= 0 && nj >= 0 && g[ni][nj] == '1') {
                    if (dsu.unite(id(i,j), id(ni,nj))) land--;
                }
            }
        }
    }
    return land;
}
```

## 加权并查集

每个节点除了 parent 还存**到根的权值**——可以维护"x 比 y 大多少"这类关系：

```cpp
int find(int x) {
    if (parent[x] == x) return x;
    int root = find(parent[x]);
    weight[x] += weight[parent[x]];   // 累加路径权
    return parent[x] = root;
}
```

经典题：LC 399 除法求值。

## 易错点

- **没初始化** `parent[i] = i` → segfault 或乱结果
- 写成 `parent[x] = y` 而不是 `parent[find(x)] = find(y)` → 错误合并
- 路径压缩**忘 `parent[x] = `** 赋值 → 没真正压缩
- 用 `find(x) == find(y)` 判同组而不是 `parent[x] == parent[y]`
- 节点数从 1 还是 0 开始要统一

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「并查集」相关关键词（并查集）的笔记，按来源分组。

### 左程云
- [[并查集|课程/左程云/002. 知识点总结/20. 算法知识点/并查集.md]]
- [[并查集代码|课程/左程云/010. 代码模板/并查集代码.md]]
- [[15 并查集及其面试题目|课程/左程云/100. 算法课程/01. 体系学习班/15 并查集及其面试题目.md]]
- [[10 并查集结构和图相关的算法|课程/左程云/100. 算法课程/10. 基础课/10 并查集结构和图相关的算法.md]]

### labuladong
- [[图拓扑排序 和 并查集|课程/labuladong/经典暴力搜索算法/图拓扑排序 和 并查集.md]]

## 参考资料

- [[../../课程/左程云/100. 算法课程]] — 左程云并查集章节
- Tarjan, *Efficiency of a Good But Not Linear Set Union Algorithm*, 1975

## 关联条目

- 上层：[[ds-overview]]
- 应用：[[mst|最小生成树]] · [[graph]]
- 兄弟：[[hash-table]]
