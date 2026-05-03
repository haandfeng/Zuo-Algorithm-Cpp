---
title: Manacher
aliases: [马拉车, 最长回文子串, Longest Palindromic Substring]
tags: [algorithm, string, palindrome]
created: 2026-05-03
updated: 2026-05-03
---

# Manacher（马拉车算法）

> **TL;DR**：求字符串中**最长回文子串**的 O(N) 算法。核心是**对称性复用**：以位置 c 为中心的回文区间 [l, r] 中，i 和 2c-i 关于 c 对称——可以借助前面已计算的对称位置的回文半径。

## 问题

朴素：每个位置都尝试**奇 / 偶**两种回文，向两边扩展，O(N²)。

Manacher：通过**统一奇偶**和**复用对称信息**，做到 O(N)。

## 统一奇偶：插入分隔符

把 `"aba"` 变成 `"^#a#b#a#$"`：

- `^` 和 `$` 是哨兵防越界
- `#` 在每两个原字符之间和首尾插入
- 结果**总长 2n + 3**，且**所有回文都变成奇数长度**

例如：

| 原串 | 处理后 | 回文中心 |
|---|---|---|
| "aba" (奇) | "^#a#b#a#$" | 'b' 位置 → 半径 3 |
| "abba" (偶) | "^#a#b#b#a#$" | 中间 '#' → 半径 4 |

## 核心算法

维护：

- `p[i]` = 以 i 为中心的最长回文半径（在新串中）
- `c` = 当前覆盖最右的回文中心
- `r` = 该回文的右端点

```cpp
string manacher_preprocess(const string& s) {
    string t = "^";
    for (char ch : s) { t += '#'; t += ch; }
    t += "#$";
    return t;
}

int longestPalindromeLen(const string& s) {
    string t = manacher_preprocess(s);
    int n = t.size();
    vector<int> p(n, 0);
    int c = 0, r = 0, ans = 0;
    for (int i = 1; i < n - 1; ++i) {
        // 关键：对称性复用
        if (i < r) p[i] = min(r - i, p[2 * c - i]);
        // 中心扩展
        while (t[i - p[i] - 1] == t[i + p[i] + 1]) p[i]++;
        // 更新最右覆盖
        if (i + p[i] > r) { c = i; r = i + p[i]; }
        ans = max(ans, p[i]);
    }
    return ans;        // 在新串中的半径 = 原串中的回文长度
}
```

## 复杂度直觉

每次中心扩展时，`r` 都**单调右移**——总扩展次数不超过 N，所以 O(N)。

## 何时用 Manacher

| 题 | 是否值得 |
|---|---|
| 求**单一**最长回文子串 | DP O(N²) 或中心扩展 O(N²) 也可，N ≤ 1000 都够 |
| N ≤ 1e5 求最长回文 | **必须** Manacher |
| 求**所有**回文子串个数 | Manacher 直接给 |
| 多次回文相关查询 | Manacher + 哈希 / 后缀数组 |

实战中 LC 5 最长回文子串用中心扩展 O(N²) 就过了——Manacher 是**算法竞赛**的标配。

## 经典题型

| 题 | 关键 |
|---|---|
| 最长回文子串 LC 5 | 中心扩展 O(N²) 或 Manacher O(N) |
| 回文子串数目 LC 647 | Manacher 求 sum of (p[i] + 1) / 2 |
| 最短回文串 LC 214 | 反串拼接 + KMP（不是 Manacher） |
| 分割回文串 LC 131 / 132 | DP + 回文判定预处理（Manacher 加速） |
| 不同的回文子串数 | Eertree (回文自动机) |

## 易错点

- 哨兵 `^` 和 `$` 必须不同于原字符（也不同于 `#`）
- `2 * c - i` 越界保护：用 `min(r - i, p[2 * c - i])`
- 答案是新串中的**半径**，对应原串中的**回文长度**
- 注意 `i - p[i] - 1` 是新串中字符位置，不会越界因为有哨兵

## 参考资料

- [[../../算法笔记/zuoAlgorithm/100. 算法课程]] — 左程云 Manacher
- 《算法竞赛进阶指南》Manacher 章节

## 关联条目

- 上层：[[../maps/algorithm-overview]]
- 兄弟：[[kmp]]
- 数据结构：[[../data-structures/trie]]（对应字符串多模式问题）
