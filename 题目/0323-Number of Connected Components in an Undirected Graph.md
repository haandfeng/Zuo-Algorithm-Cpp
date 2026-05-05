---
leetcode: 323
title: Number of Connected Components in an Undirected Graph
url: https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/
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

# 323. Number of Connected Components in an Undirected Graph

[LeetCode 链接](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)

## 来源：题单 / Blind75

dfs

```java
class Solution {
    
     private void dfs(List<Integer>[] adjList, int[] visited, int startNode) {
        visited[startNode] = 1;
         
        for (int i = 0; i < adjList[startNode].size(); i++) {
            if (visited[adjList[startNode].get(i)] == 0) {
                dfs(adjList, visited, adjList[startNode].get(i));
            }
        }
    }
    
    public int countComponents(int n, int[][] edges) {
        int components = 0;
        int[] visited = new int[n];
        
        List<Integer>[] adjList = new ArrayList[n]; 
        for (int i = 0; i < n; i++) {
            adjList[i] = new ArrayList<Integer>();
        }
        
        for (int i = 0; i < edges.length; i++) {
            adjList[edges[i][0]].add(edges[i][1]);
            adjList[edges[i][1]].add(edges[i][0]);
        }
        
        for (int i = 0; i < n; i++) {
            if (visited[i] == 0) {
                components++;
                dfs(adjList, visited, i);
            }
        }
        return components;
    }
}
```

## 来源：labuladong / 图拓扑排序 和 并查集

直接用模版就好了

```python
class UF:
    # 连通分量个数
    _count: int
    # 存储每个节点的父节点
    parent: List[int]

    # n 为图中节点的个数
    def __init__(self, n: int):
        self._count = n
        self.parent = [i for i in range(n)]

    # 将节点 p 和节点 q 连通
    def union(self, p: int, q: int):
        rootP = self.find(p)
        rootQ = self.find(q)

        if rootP == rootQ:
            return

        self.parent[rootQ] = rootP
        # 两个连通分量合并成一个连通分量
        self._count -= 1

    # 判断节点 p 和节点 q 是否连通
    def connected(self, p: int, q: int) -> bool:
        rootP = self.find(p)
        rootQ = self.find(q)
        return rootP == rootQ

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    # 返回图中的连通分量个数
    def count(self) -> int:
        return self._count
class Solution:
    def countComponents(self, n: int, edges: List[List[int]]) -> int:
        uf = UF(n)
        for p, q in edges:
            uf.union(p,q)
        return uf.count()
        
```
