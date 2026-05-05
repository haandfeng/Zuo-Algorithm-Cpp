---
leetcode: 261
title: Graph Valid Tree
url: https://leetcode.com/problems/graph-valid-tree/
difficulty: medium
tags:
  - graph
covers:
  - graph
  - topological-sort
  - union-find
sources:
  - 题单/Blind75
  - labuladong/图拓扑排序 和 并查集
date_split: 2026-05-05
solved: true
---

# 261. Graph Valid Tree

[LeetCode 链接](https://leetcode.com/problems/graph-valid-tree/)

## 来源：题单 / Blind75

很多特例没找到，给的所有参数都是有用的，忽略了n
```python
from collections import defaultdict
from typing import List

class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        # 特判：只有一个点且没有边，是一棵树
        if n == 1 and not edges:
            return True
        # 树的必要条件：边数必须是 n - 1
        if len(edges) != n - 1:
            return False

        g = defaultdict(list)
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)

        visited = set()

        def dfs(u, v):
            visited.add(v)
            for nei in g[v]:
                if nei == u:
                    continue
                if nei in visited:
                    return False     # 找到环
                if not dfs(v, nei):  # 递归结果要往外传
                    return False
            return True

        # 从 0 开始 DFS
        if not dfs(-1, 0):
            return False

        # 必须所有节点都被访问到
        return len(visited) == n
```

## 来源：labuladong / 图拓扑排序 和 并查集

给你输入编号从 0 到 n - 1 的 n 个结点，和一个无向边列表 edges（每条边用节点二元组表示），请你判断输入的这些边组成的结构是否是一棵树。

这道题也有dfs解法，和下面的思路一样。

对于添加的这条边，如果该边的两个节点本来就在同一连通分量里，那么添加这条边会产生环；反之，如果该边的两个节点不在同一连通分量里，则添加这条边不会产生环。

而判断两个节点是否连通（是否在同一个连通分量中）就是 Union-Find 算法的拿手绝活，所以这道题的解法代码如下：

```python
from typing import List

class Solution:
    # 初始化 0...n-1 共 n 个节点
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        uf = self.UF(n)
        # 遍历所有边，将组成边的两个节点进行连接
        for edge in edges:
            u = edge[0]
            v = edge[1]
            # 若两个节点已经在同一连通分量中，会产生环
            if uf.connected(u, v): # <extend down -200>![](/images/algo/kruskal/4.png) #
                return False
            # 这条边不会产生环，可以是树的一部分
            uf.union(u, v) # <extend down -150>![](/images/algo/kruskal/5.png) #
        # 要保证最后只形成了一棵树，即只有一个连通分量
        return uf.get_count() == 1

    class UF:
        # 连通分量个数
        def __init__(self, n: int):
            self.count = n
            # 存储一棵树
            self.parent = [i for i in range(n)]
            # 记录树的「重量」
            self.size = [1] * n
            # n 为图中节点的个数

        # 将节点 p 和节点 q 连通
        def union(self, p: int, q: int):
            rootP = self.find(p)
            rootQ = self.find(q)
            if rootP == rootQ:
                return

            # 小树接到大树下面，较平衡
            if self.size[rootP] > self.size[rootQ]:
                self.parent[rootQ] = rootP
                self.size[rootP] += self.size[rootQ]
            else:
                self.parent[rootP] = rootQ
                self.size[rootQ] += self.size[rootP]
            # 两个连通分量合并成一个连通分量
            self.count -= 1

        # 判断节点 p 和节点 q 是否连通
        def connected(self, p: int, q: int) -> bool:
            return self.find(p) == self.find(q)

        # 返回节点 x 的连通分量根节点
        def find(self, x: int) -> int:
            while self.parent[x] != x:
                # 进行路径压缩
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x

        # 返回图中的连通分量个数
        def get_count(self) -> int:
            return self.count
```
