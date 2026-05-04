---
title: 动态规划
aliases: [DP, Dynamic Programming, 动态规划]
tags: [pattern, dp]
created: 2026-05-03
updated: 2026-05-03
---

# 动态规划（DP）

> **TL;DR**：DP = **把问题拆成重叠子问题** + **记忆化避免重复计算**。本质是**带备忘录的递归**或**自底向上的递推**。是面试 / 竞赛中最难、最有区分度的范式。

## 五步法（左程云风格）

每道 DP 题严格按这 5 步走：

```text
1. 定义状态 dp[…] 是什么"含义"        ← 最关键，错了一切错
2. 写出初始状态（base case）
3. 写出状态转移方程  dp[i] = f(dp[j], …)
4. 决定遍历顺序：保证依赖项已经算出
5. 返回值：dp[n] 还是 max(dp)？
```

## 自顶向下 vs 自底向上

### 自顶向下（记忆化递归）

```cpp
unordered_map<int, int> memo;
int fib(int n) {
    if (n < 2) return n;
    if (memo.count(n)) return memo[n];
    return memo[n] = fib(n - 1) + fib(n - 2);
}
```

- ✅ 思维直接：直接照"递归三问"写
- ✅ 只算需要的状态
- ❌ 递归常数大，深度受限

### 自底向上（迭代填表）

```cpp
int fib(int n) {
    if (n < 2) return n;
    vector<int> dp(n + 1);
    dp[0] = 0; dp[1] = 1;
    for (int i = 2; i <= n; ++i) dp[i] = dp[i - 1] + dp[i - 2];
    return dp[n];
}
```

- ✅ 常数小、可滚动数组优化空间
- ❌ 写起来要先想清楚遍历顺序

**两种本质等价**——熟悉哪种就用哪种。

## 经典模板：5 大类

### 1. 一维线性 DP

`dp[i]` 只依赖 `dp[i-1]`、`dp[i-2]`、…：

```cpp
// 爬楼梯 LC 70
int dp[i] = dp[i - 1] + dp[i - 2];

// 打家劫舍 LC 198
int dp[i] = max(dp[i - 1], dp[i - 2] + nums[i]);

// 最长递增子序列 LC 300（O(N²) 版）
int dp[i] = 1 + max(dp[j] for j < i if nums[j] < nums[i]);
```

### 2. 二维网格 DP

`dp[i][j]` 依赖左 / 上 / 左上邻居：

```cpp
// 最小路径和 LC 64
dp[i][j] = grid[i][j] + min(dp[i-1][j], dp[i][j-1]);

// 不同路径 LC 62
dp[i][j] = dp[i-1][j] + dp[i][j-1];
```

### 3. 双串 DP（LCS / 编辑距离）

`dp[i][j]` 表示两串前 i / 前 j 字符的关系：

```cpp
// 最长公共子序列 LC 1143
if (s1[i-1] == s2[j-1]) dp[i][j] = dp[i-1][j-1] + 1;
else                    dp[i][j] = max(dp[i-1][j], dp[i][j-1]);

// 编辑距离 LC 72
if (s1[i-1] == s2[j-1]) dp[i][j] = dp[i-1][j-1];
else                    dp[i][j] = 1 + min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
```

详见 [[problems/edit-distance]]。

### 4. 背包 DP

详见 [[dp-knapsack]]。

### 5. 区间 DP

`dp[l][r]` 表示区间 `[l, r]` 的解：

```cpp
// 戳气球 LC 312
for (int len = 1; len <= n; ++len)
    for (int l = 0; l + len - 1 < n; ++l) {
        int r = l + len - 1;
        for (int k = l; k <= r; ++k)        // 最后戳的气球是 k
            dp[l][r] = max(dp[l][r], dp[l][k-1] + dp[k+1][r] + nums[l-1]*nums[k]*nums[r+1]);
    }
```

## 状态压缩 / 滚动数组

二维 DP 经常只依赖**上一行**，可以滚动到一维：

```cpp
// LC 64 最小路径和
vector<int> dp(m, INT_MAX);
dp[0] = 0;
for (int i = 0; i < n; ++i) {
    dp[0] += g[i][0];
    for (int j = 1; j < m; ++j)
        dp[j] = g[i][j] + min(dp[j], dp[j - 1]);    // dp[j] 是上一行，dp[j-1] 是当前行
}
return dp[m - 1];
```

空间从 O(NM) → O(M)。

## 何时想到 DP

| 信号 | 用法 |
|---|---|
| "求方案数 / 最大值 / 最小值 / 最长 / 最短" | 强信号 |
| "允许 / 必须 / 最多选 K 个" | 背包类 |
| "在网格 / 字符串上做最优决策" | 二维 DP |
| "区间 …" | 区间 DP |
| "树上每个子树独立求解" | [[dp-tree\|树形 DP]] |
| "状态数 ≤ 2^N，N ≤ 20" | [[dp-bitmask\|状压 DP]] |
| "数位约束 / 不超过 N 的 …" | [[dp-digit\|数位 DP]] |
| "回溯发现子问题重复" | 加备忘录 → DP |

## DP vs 贪心 vs 回溯

| 信号 | 推荐 |
|---|---|
| 最值 + 子问题重叠 | DP |
| 最值 + 子问题独立 | [[divide-and-conquer\|分治]] |
| 最值 + 局部最优可证明 | [[greedy\|贪心]] |
| 列出所有解 | [[backtracking\|回溯]] |
| 计数 / 概率 | DP |

## 经典 5 题（每个面试者必会）

| 题 | 类型 |
|---|---|
| 爬楼梯 LC 70 | 一维线性 |
| 最长递增子序列 LC 300 | 一维 / O(N²) 或 O(N log N)，见 [[problems/lis]] |
| 编辑距离 LC 72 | 双串，见 [[problems/edit-distance]] |
| 0-1 背包模板 | 见 [[dp-knapsack]] |
| 戳气球 LC 312 | 区间 |

## 易错点

- **状态定义不清晰** → 一切错
- 状态转移**漏分支**（例如 LIS 漏了"自己单独成一段"）
- 遍历顺序错 → 用了未计算的 dp 值
- base case 写错或漏写
- 滚动数组维度搞错（覆盖了还要用的数据）
- "01 背包" vs "完全背包" 维度顺序反了
- 答案是 `dp[n]` 还是 `max(dp)` 容易混

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「动态规划」相关关键词（动态规划, DP）的笔记，按来源分组。

### 灵茶山艾府和代码随想录
- [[动态规划|课程/灵茶山艾府和代码随想录/动态规划.md]]

### 左程云
- [[数位DP|课程/左程云/000.Pending/数位DP.md]]
- [[DP中建立空间感之后, 可以继续优化的题目|课程/左程云/001. 难点重点照顾/DP中建立空间感之后, 可以继续优化的题目.md]]
- [[_暴力递归改动态规划的套路|课程/左程云/001. 难点重点照顾/_暴力递归改动态规划的套路.md]]
- [[动态规划的模型总结|课程/左程云/001. 难点重点照顾/动态规划的模型总结.md]]
- [[在动态规划中, 贪心是个什么地位|课程/左程云/001. 难点重点照顾/在动态规划中, 贪心是个什么地位.md]]
- [[没有必要改成严格表结构的动态规划|课程/左程云/001. 难点重点照顾/没有必要改成严格表结构的动态规划.md]]

## 参考资料

- [[../../课程/灵茶山艾府和代码随想录/动态规划]] — 你的 DP 专题
- [[../../课程/labuladong/核心刷题框架]] — labuladong 动态规划框架
- [[../../课程/左程云/100. 算法课程]] — 左程云 DP 章节
- [[dp-roadmap]] — DP 学习路线图

## 关联条目

- 上层：[[algorithm-overview]]
- 子类：[[dp-knapsack]] · [[dp-interval]] · [[dp-tree]] · [[dp-bitmask]] · [[dp-digit]]
- 思维：[[recursion]] · [[divide-and-conquer]]
- 应用：[[problems/lis]] · [[problems/edit-distance]]
