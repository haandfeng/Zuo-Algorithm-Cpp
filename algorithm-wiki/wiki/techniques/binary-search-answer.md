---
title: 二分答案
aliases: [Binary Search the Answer, 二分答案, 最小化最大]
tags: [technique]
created: 2026-05-03
updated: 2026-05-03
---

# 二分答案

> **TL;DR**：当题目要求"**最小的最大** / **最大的最小**"且答案有**单调性**时，可以二分**答案本身**——把"求最优解"转成 N 次"判定可行性"。

## 三个识别信号

如果题目同时满足：

1. **要求最优**："最小的 X" / "最大的 Y"
2. **答案空间有界**：能猜出 `[lo, hi]`
3. **判定 check(mid) 容易**：给定一个候选答案 mid，能 O(N) 或 O(N log N) 判断"能不能达到"
4. **单调性**：如果 mid 可行，则 mid+1 也可行（或反之）

→ **二分答案**就是首选。

## 通用模板

```cpp
int binarySearchAnswer(int lo, int hi) {
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;       // 求最小：向下取整
        if (check(mid)) hi = mid;
        else            lo = mid + 1;
    }
    return lo;
}
```

或求最大：

```cpp
int binarySearchAnswerMax(int lo, int hi) {
    while (lo < hi) {
        int mid = lo + (hi - lo + 1) / 2;   // 求最大：向上取整防死循环
        if (check(mid)) lo = mid;
        else            hi = mid - 1;
    }
    return lo;
}
```

**注意 mid 的取整方向**：求最小用下取整、求最大用上取整。错了会死循环。

## 经典例 1：Koko 吃香蕉（LC 875）

题意：n 堆香蕉、第 i 堆 piles[i] 个、每小时最多吃 K 个、需在 H 小时内吃完。求最小 K。

```cpp
int minEatingSpeed(vector<int>& piles, int h) {
    auto check = [&](int K) {                      // 速度 K 是否能吃完
        long long hours = 0;
        for (int p : piles) hours += (p + K - 1) / K;     // 上取整
        return hours <= h;
    };
    int lo = 1, hi = *max_element(piles.begin(), piles.end());
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (check(mid)) hi = mid;
        else            lo = mid + 1;
    }
    return lo;
}
```

时间 O(N log max(piles))。

## 经典例 2：分割数组的最大值（LC 410）

题意：把数组分成 K 段，每段和的最大值要尽可能小。求这个最小的最大值。

```cpp
int splitArray(vector<int>& nums, int k) {
    auto check = [&](int max_sum) {                // 每段和 ≤ max_sum 时能不能 K 段内分完
        int cnt = 1, cur = 0;
        for (int x : nums) {
            if (cur + x > max_sum) { cnt++; cur = x; }
            else                     cur += x;
        }
        return cnt <= k;
    };
    int lo = *max_element(nums.begin(), nums.end());
    int hi = accumulate(nums.begin(), nums.end(), 0);
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (check(mid)) hi = mid;
        else            lo = mid + 1;
    }
    return lo;
}
```

LC 1011 在 D 天内送达包裹的能力是同型题。

## 经典例 3：H 指数 II（LC 275）

求最大的 h，使得至少有 h 篇论文被引用 ≥ h 次。

数组已排序——直接二分 h 值即可。

## 何时想到二分答案

| 信号 | 例子 |
|---|---|
| "最小的最大" | 分割数组的最大值 |
| "最大的最小" | 制作 m 束花的最少天数 |
| "在 H 时间内最少需要的速度" | Koko 吃香蕉 |
| "最少多少次切割能达到目标" | 切绳子 |
| "在 D 天内送达的最小载重" | LC 1011 |
| "最长子数组使得 …" 且无法直接滑窗 | 二分窗口长度 |

## 注意点

### 1. lo / hi 初值

- lo：物理上的下界（速度 ≥ 1、长度 ≥ 0）
- hi：物理上的上界（速度 ≤ 全部之和、长度 ≤ 数组长）

**初值取小了会错**——例如 Koko 吃香蕉 `lo = 1`（不是 0，否则除零）。

### 2. check 的方向

"能不能达到" 是 ≤、< 还是 = ？写错则二分方向反过来。

### 3. 单调性反向

某些题是"mid 越大越好满足、最大化 mid"（如最大化最小切割长度）—— 模板要切到求最大版。

### 4. 实数二分

```cpp
double lo = 0, hi = 1e9;
for (int iter = 0; iter < 100; ++iter) {
    double mid = (lo + hi) / 2;
    if (check(mid)) hi = mid;
    else            lo = mid;
}
return lo;
```

固定 100 次迭代，比看 `hi - lo < eps` 更稳。

## 经典题型

| 题 | 二分什么 |
|---|---|
| Koko 吃香蕉 LC 875 | 速度 |
| 分割数组的最大值 LC 410 | 最大段和 |
| 在 D 天内送达包裹的能力 LC 1011 | 载重 |
| 制作 m 束花所需的最少天数 LC 1482 | 天数 |
| 求出第 K 小的子数组 / 距离 LC 719 | 距离值 |
| 子数组中乘积小于 K LC 713 | 滑窗（不是二分答案） |
| 第 K 个最小的素数分数 LC 786 | 二分实数 |
| 爱吃香蕉的珂珂 | Koko |
| 最大化最小数 / 最小化最大数 | 模板 |

## 易错点

- mid 取整方向写反 → 死循环
- check 函数边界写错（特别是上取整 `(a + b - 1) / b`）
- lo / hi 初值不够覆盖最优解
- check 中算溢出（多个 long long 相乘 / 加大）
- 单调性方向搞错——画一下 check 真值表

## 参考资料

- [[../../课程/灵茶山艾府 + 代码随想录/二分查找]] — 你的二分专题（含二分答案）
- [[../../课程/labuladong/核心刷题框架]] — labuladong 二分答案

## 关联条目

- 上层：[[algorithm-overview]]
- 兄弟：[[binary-search]] · [[greedy]]
- 应用：[[sliding-window]]
