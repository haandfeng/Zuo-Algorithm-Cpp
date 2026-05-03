---
title: 最长递增子序列
aliases: [LIS, Longest Increasing Subsequence, LC 300]
tags: [problem, dp, binary-search, classic]
difficulty: medium
leetcode: 300
created: 2026-05-03
updated: 2026-05-03
---

# 最长递增子序列（LC 300）

> **题号**：LeetCode 300
> **难度**：Medium
> **核心**：DP O(N²) 入门 + **耐心排序 + [[../algorithms/binary-search|二分]]** 优化到 O(N log N)。**面试 DP 必考**。

## 题目

给定无序整数数组 `nums`，找出**最长严格递增子序列**的长度。

子序列 ≠ 子数组——子序列**可以不连续**，但保持原顺序。

```text
nums = [10,9,2,5,3,7,101,18]
LIS  = [2,5,7,101] → 长度 4
```

## 解法 1：DP O(N²)

`dp[i]` 表示**以 `nums[i]` 结尾**的 LIS 长度。

```cpp
int lengthOfLIS(vector<int>& nums) {
    int n = nums.size();
    vector<int> dp(n, 1);             // 自己单独成一段
    for (int i = 1; i < n; ++i)
        for (int j = 0; j < i; ++j)
            if (nums[j] < nums[i])
                dp[i] = max(dp[i], dp[j] + 1);
    return *max_element(dp.begin(), dp.end());
}
```

要点：
- 答案不是 `dp[n-1]`，而是 `max(dp)` —— 最长子序列不一定以最后一个结尾
- 严格递增用 `<`，允许相等用 `≤`

时间 O(N²)、空间 O(N)。

## 解法 2：耐心排序 + 二分 O(N log N)

维护一个数组 `tails`，`tails[k]` 表示**长度为 k+1 的所有递增子序列中，结尾最小的那个值**。

```cpp
int lengthOfLIS(vector<int>& nums) {
    vector<int> tails;
    for (int x : nums) {
        auto it = lower_bound(tails.begin(), tails.end(), x);
        if (it == tails.end()) tails.push_back(x);
        else                   *it = x;
    }
    return tails.size();
}
```

时间 O(N log N)、空间 O(N)。

### 直觉：扑克牌"耐心排序"

把每张牌（数字）放到第一个**比它大的堆顶**上；如果没有就开新堆。最后**堆数 = LIS 长度**。

```text
nums = [10, 9, 2, 5, 3, 7, 101, 18]

10  → [10]                        1 堆
 9  → 9 < 10，覆盖 → [9]          1 堆
 2  → 2 < 9，覆盖 → [2]            1 堆
 5  → 5 > 2，新堆 → [2, 5]         2 堆
 3  → 3 ≤ 5，覆盖 → [2, 3]         2 堆
 7  → 7 > 3，新堆 → [2, 3, 7]      3 堆
101 → 新堆 → [2, 3, 7, 101]        4 堆
18  → 18 ≤ 101，覆盖 → [2,3,7,18]  4 堆
```

最终长度 4 即答案。注意 `tails` 内**不是**实际的 LIS，但**长度是对的**。

### 严格递增 vs 非严格递增

- **严格递增**（< ）→ `lower_bound`（找第一个 ≥ x 的位置覆盖）
- **非严格递增**（≤）→ `upper_bound`（找第一个 > x 的位置覆盖）

## 输出 LIS 本身

要输出**实际的子序列**，需要在 DP 时记录前驱：

```cpp
vector<int> prev(n, -1);
for (int i = 1; i < n; ++i)
    for (int j = 0; j < i; ++j)
        if (nums[j] < nums[i] && dp[j] + 1 > dp[i]) {
            dp[i] = dp[j] + 1;
            prev[i] = j;
        }

// 找 dp 最大的下标 k，沿 prev 回溯
vector<int> path;
int k = max_element(dp.begin(), dp.end()) - dp.begin();
while (k != -1) { path.push_back(nums[k]); k = prev[k]; }
reverse(path.begin(), path.end());
```

## 变体题

| 题 | 变化 |
|---|---|
| 最长递增子序列的个数 LC 673 | 同时维护 cnt[i] |
| 俄罗斯套娃信封问题 LC 354 | 二维：先排序 + LIS |
| 最长摆动序列 LC 376 | 不是 LIS，DP / 贪心 |
| 最长递增子序列输出 | 见上 |
| 拼接最大数 / 装箱 | LIS 变形 |
| 堆叠箱子 | 三维 LIS |
| 最长上升子序列 II（数据结构题） | [[../data-structures/segment-tree\|线段树]] 优化 DP |

## 推广：LCS / 编辑距离

LIS 是**单串 DP**，扩展到双串就是 LCS（最长公共子序列）和编辑距离（LC 72，详见 [[edit-distance]]）。

## 易错点

- 答案是 `max(dp)` 不是 `dp[n-1]`
- 二分版本中要用 `lower_bound` （严格递增）还是 `upper_bound`（≤）
- "tails 内不是实际 LIS" —— 别试图直接打印 tails 当答案
- 边界：n=0 返回 0，n=1 返回 1
- 严格 / 非严格容易混

## 在你 Vault 中的位置

- [[../../题单/Hot100|Hot100]]
- [[../../课程/灵茶山艾府 + 代码随想录/动态规划]] DP 章节

## 关联条目

- 范式：[[../patterns/dp]] · [[../patterns/dp-roadmap]]
- 算法：[[../algorithms/binary-search]]
- 兄弟题：[[edit-distance]]
- 概念：[[../concepts/divide-and-conquer]]（耐心排序）
