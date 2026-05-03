---
title: 单词接龙
aliases: [Word Ladder, LC 127]
tags: [problem, bfs, classic]
difficulty: hard
leetcode: 127
created: 2026-05-03
updated: 2026-05-03
---

# 单词接龙（LC 127）

> **题号**：LeetCode 127
> **难度**：Hard
> **核心**：**[[../algorithms/bfs-dfs|BFS]] 求最少步数** 的范例题。**双向 BFS** 是优化的灵魂。

## 题目

给两个单词 `beginWord` 和 `endWord`、词典 `wordList`。每次只能改一个字母，要求每次变换后的新单词必须在词典中。求**从 beginWord 到 endWord 的最短变换序列长度**（不存在则返回 0）。

```text
beginWord = "hit", endWord = "cog"
wordList  = ["hot","dot","dog","lot","log","cog"]
最短路径：hit → hot → dot → dog → cog,  长度 5
```

## 抽象成图

每个**单词是节点**，**两个单词只差一个字母**就有边相连。问题变成"无权图最短路"——天然 BFS。

但**直接建图是 O(N²·L)**——双重循环比对每对单词。当词典大时不行。

## 关键技巧：通配符建图

把每个单词扩展成 L 个**通配模式**：例如 `dog → *og, d*g, do*`。两个单词共享某个模式 → 它们差一个字母。

```cpp
unordered_map<string, vector<string>> patterns;
for (string& w : wordList) {
    for (int i = 0; i < (int)w.size(); ++i) {
        string p = w.substr(0, i) + "*" + w.substr(i + 1);
        patterns[p].push_back(w);
    }
}
```

时间 O(N·L²)，空间 O(N·L²)。

## 解法 1：单向 BFS

```cpp
int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
    unordered_set<string> dict(wordList.begin(), wordList.end());
    if (!dict.count(endWord)) return 0;
    int L = beginWord.size();

    queue<string> q;
    q.push(beginWord);
    unordered_set<string> visited{beginWord};
    int step = 1;

    while (!q.empty()) {
        int sz = q.size();
        while (sz--) {
            string w = q.front(); q.pop();
            if (w == endWord) return step;
            for (int i = 0; i < L; ++i) {
                string nw = w;
                for (char c = 'a'; c <= 'z'; ++c) {
                    if (c == w[i]) continue;
                    nw[i] = c;
                    if (dict.count(nw) && !visited.count(nw)) {
                        visited.insert(nw);
                        q.push(nw);
                    }
                }
            }
        }
        step++;
    }
    return 0;
}
```

时间 O(N·L²·26)。

## 解法 2：双向 BFS（推荐）

从 `beginWord` 和 `endWord` **同时往中间扩**——相遇即终止。

把搜索空间从 $b^d$ 降到 $2 b^{d/2}$，对长链路是数量级提升。

```cpp
int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
    unordered_set<string> dict(wordList.begin(), wordList.end());
    if (!dict.count(endWord)) return 0;
    int L = beginWord.size();

    unordered_set<string> q1{beginWord}, q2{endWord};
    unordered_set<string> visited;
    int step = 1;

    while (!q1.empty() && !q2.empty()) {
        if (q1.size() > q2.size()) swap(q1, q2);     // 总是扩较小那边
        unordered_set<string> next;
        for (auto& w : q1) {
            string nw = w;
            for (int i = 0; i < L; ++i) {
                char old = nw[i];
                for (char c = 'a'; c <= 'z'; ++c) {
                    if (c == old) continue;
                    nw[i] = c;
                    if (q2.count(nw)) return step + 1;       // 相遇！
                    if (dict.count(nw) && !visited.count(nw)) {
                        visited.insert(nw);
                        next.insert(nw);
                    }
                }
                nw[i] = old;
            }
        }
        q1 = next;
        step++;
    }
    return 0;
}
```

要点：
- **每次扩较小的那一侧**——最优
- 用 `unordered_set` 而不是 queue，便于检查"相遇"
- 找到就 `step + 1`（两侧步数 + 这一步连接）

## 双向 BFS 的省力直觉

设答案路径长 d、每步扩展 b 个邻居：

- 单向 BFS 总状态数 ~ $b^d$
- 双向 BFS 两侧各扩 d/2 → 总状态数 ~ $2 b^{d/2}$

例如 b=10、d=6：$10^6$ vs $2 × 10^3$ —— **1000 倍**差距。

## 变体题

| 题 | 变化 |
|---|---|
| 单词接龙 II LC 126 | 输出**所有**最短变换序列（BFS + 路径回溯） |
| 最小基因变化 LC 433 | 同结构（4 字符基因 ATCG） |
| 解开钥匙的最少次数 LC 752 | 双向 BFS 经典 |
| 滑动谜题 LC 773 | 状态空间 BFS |
| 最短桥 LC 934 | 多源 BFS |

### LC 126 输出所有最短路径

需要：

1. **正向 BFS** 建分层信息（每个单词的最短距离）
2. **反向 DFS** 沿"距离严格 -1"的方向回溯，搜出所有路径

```cpp
unordered_map<string, vector<string>> graph;       // u → 所有"distance(u) = distance(v) + 1"的 v
// BFS 填 graph
// DFS 从 endWord 回溯
```

代码较长，是经典"BFS + DFS 配合"题型。

## 易错点

- BFS 没**入队时标记 visited** → TLE（同一节点多次入队）
- 双向 BFS 忘记**总扩较小一侧** → 退化成单向
- 通配符 `*` 在多个单词长度不一时要小心
- LC 126 输出路径时，**不能先访问就在 visited 加**——多条最短路径会被漏掉

## 在你 Vault 中的位置

- 列在 [[../../题单/Hot100|Hot100]] 中（图 / BFS 类）
- [[../../课程/labuladong/经典暴力搜索算法]] BFS 章节

## 关联条目

- 算法：[[../algorithms/bfs-dfs]]
- 数据结构：[[../data-structures/queue]] · [[../data-structures/hash-table]]
- 兄弟题：[[lis]]（不相关，但都是经典题）
