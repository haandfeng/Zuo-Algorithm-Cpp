---
title: 编辑距离
aliases: [Edit Distance, Levenshtein, LC 72]
tags: [problem, dp, classic]
difficulty: medium
leetcode: 72
created: 2026-05-03
updated: 2026-05-03
---

# 编辑距离（LC 72）

> **题号**：LeetCode 72
> **难度**：Medium（其实接近 Hard）
> **核心**：**双串 DP** 的范式题。学会它，LCS、最长公共子串、最短公共超序列都会。

## 题目

给定两个字符串 `word1` 和 `word2`，返回**把 word1 变成 word2 所需的最少操作数**。

允许的操作：

1. 插入一个字符
2. 删除一个字符
3. 替换一个字符

```text
word1 = "horse",  word2 = "ros"
答案 = 3：
   horse → rorse（替换 h→r）
   rorse → rose（删除 r）
   rose  → ros（删除 e）
```

## DP 状态定义

`dp[i][j]` = 把 `word1[0..i-1]` 变成 `word2[0..j-1]` 的最少操作数。

**关键**：`i, j` 是**长度**而不是**下标**——这样 base case `dp[0][j] = j`（空串变成长度 j 的串需要 j 次插入）写起来简洁。

## 状态转移

```text
若 word1[i-1] == word2[j-1]：
    dp[i][j] = dp[i-1][j-1]                          // 不需要操作

否则三选一：
    dp[i][j] = 1 + min(
        dp[i-1][j],      // 删除 word1[i-1]
        dp[i][j-1],      // 在 word1 末尾插入 word2[j-1]
        dp[i-1][j-1]     // 替换 word1[i-1] 为 word2[j-1]
    )
```

## 实现

```cpp
int minDistance(string w1, string w2) {
    int n = w1.size(), m = w2.size();
    vector<vector<int>> dp(n + 1, vector<int>(m + 1));
    for (int i = 0; i <= n; ++i) dp[i][0] = i;          // word1 → 空串：i 次删除
    for (int j = 0; j <= m; ++j) dp[0][j] = j;          // 空串 → word2：j 次插入
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= m; ++j) {
            if (w1[i-1] == w2[j-1])
                dp[i][j] = dp[i-1][j-1];
            else
                dp[i][j] = 1 + min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
        }
    }
    return dp[n][m];
}
```

时间 O(NM)、空间 O(NM)。

## 滚动数组优化空间

`dp[i][j]` 只依赖 `dp[i-1][...]` 和 `dp[i][j-1]` —— 可以滚动到 O(min(N, M))：

```cpp
int minDistance(string w1, string w2) {
    int n = w1.size(), m = w2.size();
    if (n < m) { swap(w1, w2); swap(n, m); }     // 让 m 是较小的
    vector<int> dp(m + 1);
    iota(dp.begin(), dp.end(), 0);                // dp[j] = j
    for (int i = 1; i <= n; ++i) {
        int prev = dp[0];                          // 保存左上角
        dp[0] = i;
        for (int j = 1; j <= m; ++j) {
            int tmp = dp[j];
            if (w1[i-1] == w2[j-1])
                dp[j] = prev;
            else
                dp[j] = 1 + min({dp[j], dp[j-1], prev});
            prev = tmp;
        }
    }
    return dp[m];
}
```

空间 O(min(N, M))。**注意 `prev` 的维护**——它保存左上角的旧值。

## 三个操作的几何直觉

`dp[i-1][j]`：上方
`dp[i][j-1]`：左方
`dp[i-1][j-1]`：左上方

```text
    "" r o s
""  0  1 2 3
h   1  1 2 3
o   2  2 1 2
r   3  2 2 2
s   4  3 3 2
e   5  4 4 3   ← 答案
```

每个格子从上 / 左 / 左上**取最小 + 1**（除非字符相同直接抄左上）。

## 变体题

| 题 | 变化 |
|---|---|
| 最长公共子序列 LC 1143 | 状态相同，转移略不同 |
| 不同的子序列 LC 115 | 求方案数 |
| 删除两个字符串使相同 LC 583 | 等价于 (n + m - 2 * LCS) |
| 通配符匹配 LC 44 | `?` 和 `*` 的 DP |
| 正则匹配 LC 10 | `.` 和 `*`，转移更复杂 |
| 交错字符串 LC 97 | 双串布尔 DP |
| 最长公共子串 | 子串而非子序列：转移条件不同 |
| 字符串多次插入 / 删除使相等的最小代价（带权） | 同模板，只是 +1 换成 +cost |
| Hunt-Szymanski 算法 | 输出 LCS 的实际字符串 |

## 输出操作序列

回溯 DP 矩阵：

```cpp
void traceback(string& w1, string& w2, vector<vector<int>>& dp) {
    int i = w1.size(), j = w2.size();
    vector<string> ops;
    while (i > 0 && j > 0) {
        if (w1[i-1] == w2[j-1]) { i--; j--; }
        else if (dp[i][j] == dp[i-1][j-1] + 1) {
            ops.push_back("replace " + string(1, w1[i-1]) + " with " + w2[j-1]);
            i--; j--;
        } else if (dp[i][j] == dp[i-1][j] + 1) {
            ops.push_back("delete " + string(1, w1[i-1]));
            i--;
        } else {
            ops.push_back("insert " + string(1, w2[j-1]));
            j--;
        }
    }
    while (i > 0) { ops.push_back("delete " + string(1, w1[--i])); }
    while (j > 0) { ops.push_back("insert " + string(1, w2[--j])); }
    reverse(ops.begin(), ops.end());
}
```

## 易错点

- 状态索引混淆：`dp[i][j]` 中 `i, j` 是**长度**还是**下标**？两种写法都行但**全程统一**
- base case `dp[0][j]` 和 `dp[i][0]` 写成 0 或 1 → 错
- 替换的 `+1` 和 `dp[i-1][j-1]` 顺序写反
- 空间优化时 `prev` 没正确维护
- 字符相同时**还加 1** → 多算

## 在你 Vault 中的位置

- [[../../题单/Hot100|Hot100]] 必考
- [[../../课程/灵茶山艾府 + 代码随想录/动态规划]] DP 双串章节
- [[../../课程/labuladong/核心刷题框架]] — labuladong 编辑距离

## 关联条目

- 范式：[[../patterns/dp]] · [[../patterns/dp-roadmap]]
- 兄弟题：[[lis]]（单串 DP）
- 字符串：[[../algorithms/kmp]]
