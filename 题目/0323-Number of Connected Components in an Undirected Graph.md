---
leetcode: 323
title: Number of Connected Components in an Undirected Graph
url: https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/
difficulty: medium
tags: []
covers: []
sources:
  - 题单/Blind75
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
