---
leetcode: 261
title: Graph Valid Tree
url: https://leetcode.com/problems/graph-valid-tree/
difficulty: medium
tags: []
covers: []
sources:
  - 题单/Blind75
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
