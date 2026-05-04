---
leetcode: 1245
title: Tree Diameter
url: https://leetcode.com/problems/tree-diameter/
difficulty: 
tags:
  - dp
covers:
  - dp
sources:
  - 灵茶山艾府和代码随想录/动态规划
date_split: 2026-05-05
solved: true
---

# 1245. Tree Diameter

[LeetCode 链接](https://leetcode.com/problems/tree-diameter/)

## 来源：灵茶山艾府和代码随想录 / 动态规划

一道会员题，树的直径。我在国际站有会员，0x3f说和[[#[2246. 相邻字符不同的最长路径](https //leetcode.cn/problems/longest-path-with-different-adjacent-characters/)]]很像，于是做一下

需要注意这道题没有给定根节点，所以要判断是否重复访问，且要双向都加入。并且这个是
```python
from typing import List

class Solution:
    def treeDiameter(self, edges: List[List[int]]) -> int:
        if not edges:
            return 0  # 只有一个点，直径为 0

        # 更稳妥：按标签算节点数（避免坏输入）
        n = max(max(a, b) for a, b in edges) + 1

        g = [[] for _ in range(n)]
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)

        self.ans = 0

        def dfs(u: int, p: int) -> int:
            # 返回从 u 往下的最长链长
            top1 = 0  # 第一长的向下链
            top2 = 0  # 第二长的向下链
            for v in g[u]:
                if v == p:  # 避免回到父亲
                    continue
                d = dfs(v, u) + 1
                if d > top1:
                    top2 = top1
                    top1 = d
                elif d > top2:
                    top2 = d
            # 经过 u 的最长路径 = top1 + top2
            self.ans = max(self.ans, top1 + top2)
            return top1

        dfs(0, -1)
        return self.ans
```
