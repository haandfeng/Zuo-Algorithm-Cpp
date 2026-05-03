---
title: 背包 DP
aliases: [Knapsack, 背包问题, 01 背包, 完全背包]
tags: [pattern, dp]
created: 2026-05-03
updated: 2026-05-03
---

# 背包 DP

> **TL;DR**：背包问题 = 给定容量 W 和 N 个物品，**选出物品**使得**总重量 ≤ W** 且**价值最大**（或满足某条件）。**01 背包**（每个物品 0/1 个）、**完全背包**（每个物品无限）、**多重背包**（每个有限）、**分组背包**（互斥分组）。

## 0-1 背包（每个物品 0/1 次）

### 二维状态

`dp[i][j]` = "从前 i 个物品里选、容量不超过 j 的最大价值"。

```cpp
int knapsack01(int W, vector<int>& weights, vector<int>& values) {
    int n = weights.size();
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));
    for (int i = 1; i <= n; ++i) {
        for (int j = 0; j <= W; ++j) {
            dp[i][j] = dp[i - 1][j];                    // 不选第 i 个
            if (j >= weights[i - 1])
                dp[i][j] = max(dp[i][j],
                               dp[i - 1][j - weights[i - 1]] + values[i - 1]);   // 选第 i 个
        }
    }
    return dp[n][W];
}
```

时间 O(NW)、空间 O(NW)。

### 滚动到一维

`dp[j]` 直接覆盖前一行——但**容量必须从大到小**遍历，否则同一物品会被选多次：

```cpp
vector<int> dp(W + 1, 0);
for (int i = 0; i < n; ++i) {
    for (int j = W; j >= weights[i]; --j) {            // ← 倒序！
        dp[j] = max(dp[j], dp[j - weights[i]] + values[i]);
    }
}
return dp[W];
```

空间 O(W)。

## 完全背包（每个物品无限次）

只需把 0-1 背包的内层循环改成**正序**：

```cpp
vector<int> dp(W + 1, 0);
for (int i = 0; i < n; ++i) {
    for (int j = weights[i]; j <= W; ++j) {            // ← 正序！
        dp[j] = max(dp[j], dp[j - weights[i]] + values[i]);
    }
}
```

**为什么正序就变成完全背包**：正序时 `dp[j - weights[i]]` 已经包含"选过第 i 个物品的状态"，再选一次就是"选了多次"。

## 多重背包（每个物品有限 k_i 次）

### 朴素：每件物品当作独立 0-1 物品

```cpp
for (int i = 0; i < n; ++i)
    for (int c = 1; c <= count[i]; ++c)
        for (int j = W; j >= weights[i]; --j)
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i]);
```

时间 O(W · ΣK)。

### 二进制优化

把 K_i 个第 i 件物品**二进制分组**为 1, 2, 4, ..., (剩余) 共 log K_i 组，每组当一个 0-1 物品：

```cpp
vector<pair<int,int>> items;        // {weight, value}
for (int i = 0; i < n; ++i) {
    int k = count[i];
    for (int c = 1; c <= k; c <<= 1) {
        items.push_back({c * weights[i], c * values[i]});
        k -= c;
    }
    if (k > 0) items.push_back({k * weights[i], k * values[i]});
}
// 然后跑标准 0-1 背包
```

时间 O(W · N log K)。

### 单调队列优化

最优 O(NW)，但代码复杂——竞赛常用。

## 分组背包

每组**只能选一个**物品：

```cpp
for (int g = 0; g < groups; ++g) {
    for (int j = W; j >= 0; --j) {                    // 倒序：每组只能选 0 / 1 个
        for (auto& [w, v] : group_items[g]) {
            if (j >= w) dp[j] = max(dp[j], dp[j - w] + v);
        }
    }
}
```

注意**外层枚举组**、**中层枚举容量倒序**、**内层枚举组内物品** —— 顺序不能换。

## 经典变形

| 题 | 类型 | 关键点 |
|---|---|---|
| 分割等和子集 LC 416 | 0-1 背包 | 求 `sum/2` 是否可达 |
| 目标和 LC 494 | 0-1 背包 | 转化为 `(sum + target) / 2` |
| 一和零 LC 474 | 二维 0-1 背包 | 两个容量维度 |
| 零钱兑换 LC 322 | 完全背包 | 求最少硬币数 |
| 零钱兑换 II LC 518 | 完全背包 | 求方案数 |
| 单词拆分 LC 139 | 完全背包 | 字符串版 |
| 完全平方数 LC 279 | 完全背包 | 求最少完全平方数和为 N |
| 组合总和 IV LC 377 | 完全背包 | 求**排列数**——内外循环互换 |

## 求方案数 vs 求最大值

| 求 | 转移 | 初值 |
|---|---|---|
| 最大价值 | `dp[j] = max(dp[j], dp[j-w] + v)` | 通常 0 |
| 是否可达 | `dp[j] = dp[j] || dp[j-w]` | `dp[0] = true` |
| 方案数 | `dp[j] = dp[j] + dp[j-w]` | `dp[0] = 1` |
| 最小数量 | `dp[j] = min(dp[j], dp[j-w] + 1)` | `dp[0] = 0`，其它 `INT_MAX` |

## 求组合数 vs 求排列数

**关键差别**：内外循环顺序。

```cpp
// 组合数（顺序不算）：外物品、内容量
for (int i = 0; i < n; ++i)
    for (int j = w[i]; j <= W; ++j)
        dp[j] += dp[j - w[i]];

// 排列数（顺序算）：外容量、内物品
for (int j = 0; j <= W; ++j)
    for (int i = 0; i < n; ++i)
        if (j >= w[i]) dp[j] += dp[j - w[i]];
```

LC 377 组合总和 IV 是排列；LC 518 零钱兑换 II 是组合。

## 速记口诀

```text
0-1 背包：     倒序             dp[j] = max(dp[j], dp[j-w] + v)
完全背包：     正序             dp[j] = max(dp[j], dp[j-w] + v)
多重背包：     拆成二进制 0-1 组
分组背包：     外组、中容量倒序、内物品
求最大价值：   max
求是否可达：   ||
求方案数：     +
求最小数量：   min
求组合：       外物品内容量
求排列：       外容量内物品
```

## 易错点

- 0-1 背包内层**没倒序** → 物品被多次选，变成了完全背包
- 完全背包**用倒序** → 退化成 0-1 背包
- 求方案数 `dp[0]` 必须是 1，不是 0
- 求最小数量初值 `INT_MAX` 时转移要防溢出
- 二维降一维后**没复用空间** → 浪费
- 求"恰好等于" 而不是"≤" → base case 不一样

## 参考资料

- [[../../课程/灵茶山艾府 + 代码随想录/动态规划]] — 你的 DP 专题（含背包）
- [[../../课程/左程云/100. 算法课程]] — 左程云背包章节
- 《背包九讲》by dd_engi（经典）

## 关联条目

- 上层：[[dp]] · [[dp-roadmap]]
- 兄弟：[[dp-interval]] · [[dp-tree]] · [[dp-bitmask]]
- 思维：[[divide-and-conquer]] · [[recursion]]
