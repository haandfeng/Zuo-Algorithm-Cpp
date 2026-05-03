---
title: 状压 DP
aliases: [Bitmask DP, 状态压缩 DP]
tags: [pattern, dp]
created: 2026-05-03
updated: 2026-05-03
---

# 状压 DP

> **TL;DR**：当"状态"是**一个集合**且**集合元素 ≤ ~20** 时，用一个 int 的二进制位表示这个集合，整个 DP 的状态数是 $2^N$。是 [[dp]] + [[../algorithms/bit-manipulation|位运算]] 的合体。

## 核心思想

把"我已经选了 / 访问了哪些元素"这个集合压成一个整数。例如 N=4：

```text
集合 {0, 2}     → 0b0101 = 5
集合 {1, 3}     → 0b1010 = 10
全集合           → 0b1111 = 15
```

`dp[mask][...]` 表示"已经处理了 mask 标记的元素时的状态"。

## 经典题：旅行商问题（TSP）

N 个城市、距离矩阵 d，求经过每个城市恰好一次再回起点的最短路。

```cpp
int tsp(vector<vector<int>>& d) {
    int n = d.size();
    vector<vector<int>> dp(1 << n, vector<int>(n, INT_MAX / 2));
    dp[1][0] = 0;          // 从城市 0 出发，已访问 {0}
    for (int mask = 1; mask < (1 << n); ++mask) {
        for (int u = 0; u < n; ++u) {
            if (!(mask & (1 << u))) continue;        // u 必须在 mask 中
            if (dp[mask][u] == INT_MAX / 2) continue;
            for (int v = 0; v < n; ++v) {
                if (mask & (1 << v)) continue;       // v 不能已经访问过
                int nm = mask | (1 << v);
                dp[nm][v] = min(dp[nm][v], dp[mask][u] + d[u][v]);
            }
        }
    }
    int ans = INT_MAX;
    for (int u = 1; u < n; ++u)
        ans = min(ans, dp[(1 << n) - 1][u] + d[u][0]);
    return ans;
}
```

复杂度 O(2^N · N²)，N=20 时约 4 × 10^8，刚好能跑。

## 何时想到状压 DP

| 信号 | 例子 |
|---|---|
| N ≤ 20 且涉及"子集选 / 不选" | 旅行商、棋盘覆盖 |
| 状态是"已使用的元素集合" | TSP、最短哈密顿路径 |
| 棋盘 / 网格 + 行内选择 | 玉米田、互不侵占 |

## 子集枚举

枚举 `mask` 的所有子集：

```cpp
for (int sub = mask; sub > 0; sub = (sub - 1) & mask) {
    // sub 是 mask 的非空子集
}
```

复杂度：所有 mask 的子集总和是 $3^N$，比 $4^N$ 优。

## 经典题型

| 题 | 用法 |
|---|---|
| 旅行商问题 | TSP 模板 |
| 互不侵占的国王 / 玉米田（POJ） | 行内 mask + 行间转移 |
| 划分为 K 个相等子集 LC 698 | mask 标记已用 + 当前桶余量 |
| 火柴拼正方形 LC 473 | 同上 |
| 集合的最大相等划分 | 子集枚举 |
| 优惠券 LC 1928 | 状压 + 多维 |

## 易错点

- N > 20 → 状态 > 1M，可能不够内存
- 位运算优先级！`mask & (1 << i) == 0` 错——要 `(mask & (1 << i)) == 0`
- `(1 << n)` 在 n=31 时 UB → 用 `1LL << n`
- 子集枚举时 `sub = 0` 不在循环里（条件 `sub > 0` 需另行处理空集）
- 状压本质是**集合 → 整数**，写转移时**画图**避免混乱

## 参考资料

- [[../../课程/左程云/100. 算法课程]] — 左程云状压 DP 章节
- 《算法竞赛进阶指南》状压 DP 章节

## 关联条目

- 上层：[[dp]] · [[dp-roadmap]]
- 工具：[[../algorithms/bit-manipulation]]
- 兄弟：[[dp-tree]] · [[dp-interval]] · [[dp-digit]]
