---
title: 区间 DP
aliases: [Interval DP]
tags: [pattern, dp]
created: 2026-05-03
updated: 2026-05-03
---

# 区间 DP

> **TL;DR**：状态是 `dp[l][r]` = "区间 `[l, r]` 的最优解"，转移枚举**最后操作的位置 k**。复杂度通常 O(N³)。

## 通用模板

```cpp
// 按区间长度从小到大枚举
for (int len = 1; len <= n; ++len) {
    for (int l = 0; l + len - 1 < n; ++l) {
        int r = l + len - 1;
        if (l == r) { dp[l][r] = base; continue; }
        for (int k = l; k < r; ++k) {        // 枚举分割 / 最后一个操作
            dp[l][r] = combine(dp[l][r], dp[l][k], dp[k+1][r], cost(l, k, r));
        }
    }
}
```

**关键**：必须**先算短区间**——保证 `dp[l][k]` 和 `dp[k+1][r]` 都已经有值。

## 经典题：戳气球（LC 312）

```cpp
int maxCoins(vector<int>& nums) {
    int n = nums.size();
    vector<int> v(n + 2);
    v[0] = v[n + 1] = 1;
    for (int i = 0; i < n; ++i) v[i + 1] = nums[i];

    vector<vector<int>> dp(n + 2, vector<int>(n + 2, 0));
    for (int len = 1; len <= n; ++len) {
        for (int l = 1; l + len - 1 <= n; ++l) {
            int r = l + len - 1;
            for (int k = l; k <= r; ++k) {       // k 是 [l, r] 中**最后**戳的气球
                dp[l][r] = max(dp[l][r],
                               dp[l][k-1] + v[l-1] * v[k] * v[r+1] + dp[k+1][r]);
            }
        }
    }
    return dp[1][n];
}
```

**关键反向思维**：枚举"最后戳的气球" k，这样 `[l, k-1]` 和 `[k+1, r]` 已被戳完，k 两侧的"邻居"是 `v[l-1]` 和 `v[r+1]`（不是动态变化的）。

## 经典题：合并石头（LC 1000）

K 堆相邻石头一次合并。求合并所有为一堆的最小代价。

转移：`dp[l][r][k]` 表示把 `[l, r]` 合并成 k 堆的最小代价。

## 何时想到区间 DP

| 信号 | 例子 |
|---|---|
| "区间 [l, r] 的最优 / 计数" | 戳气球、矩阵链乘 |
| "把一段切若干段" | 合并石头 |
| "回文子序列 / 字符串变换" | LC 516 最长回文子序列 |
| "括号 / 括号匹配 + 计数" | 加括号最小化 |

## 经典题型

| 题 | 关键 |
|---|---|
| 戳气球 LC 312 | 反向枚举最后戳 |
| 合并石头 LC 1000 | 三维 DP |
| 矩阵链乘 | 经典 O(N³) |
| 最长回文子序列 LC 516 | dp[l][r] = …|
| 加括号的不同方式 LC 241 | 区间分割 |
| 切棍子的最小成本 LC 1547 | 排序 + 区间 DP |
| 移除盒子 LC 546 | 三维区间 DP |

## 复杂度优化

朴素 O(N³)。对**满足四边形不等式**的题型可以优化到 O(N²)（Knuth 优化）。算法竞赛进阶题。

## 易错点

- 枚举顺序：**先长度后左端**——不能反
- k 的范围：`l <= k < r` 还是 `l < k <= r`，画图确认
- 边界 `l == r`（长度 1）的初值
- 戳气球类问题"枚举最后操作"是反直觉的，背模板
- 多维区间 DP 数组开足够大

## 参考资料

- [[../../课程/左程云/100. 算法课程]] — 左程云区间 DP 章节
- 《算法竞赛进阶指南》区间 DP 章节

## 关联条目

- 上层：[[dp]] · [[dp-roadmap]]
- 兄弟：[[dp-knapsack]] · [[dp-tree]] · [[dp-bitmask]]
