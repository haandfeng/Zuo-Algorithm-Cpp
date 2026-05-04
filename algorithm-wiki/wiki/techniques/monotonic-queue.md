---
title: 单调队列
aliases: [Monotonic Queue, 单调双端队列]
tags: [technique, queue]
created: 2026-05-03
updated: 2026-05-03
---

# 单调队列

> **TL;DR**：单调队列 = 在 [[queue|双端队列 deque]] 中维护一个**单调序列**。专治"**滑动窗口最值**"，O(N)。

## 与单调栈的区别

| 维度 | [[monotonic-stack\|单调栈]] | 单调队列 |
|---|---|---|
| 底层 | 栈 / vector | **deque（两端可弹）** |
| 解决 | "下一个更大 / 更小" | "**滑窗最大 / 最小**" |
| 删除 | 只在一端 | **队首过期、队尾失效** |

单调队列**两端都要弹**，所以必须用 deque。

## 经典模板：滑窗最大值

```cpp
// LC 239 滑动窗口最大值
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    deque<int> dq;                      // 存下标，对应值递减
    vector<int> ans;
    for (int i = 0; i < (int)nums.size(); ++i) {
        // 1. 队尾失效：保持队列从前到后单调递减
        while (!dq.empty() && nums[dq.back()] <= nums[i]) dq.pop_back();
        dq.push_back(i);

        // 2. 队首过期：超出窗口左边界
        if (dq.front() <= i - k) dq.pop_front();

        // 3. 收集答案：窗口大小到了 k 才开始
        if (i >= k - 1) ans.push_back(nums[dq.front()]);
    }
    return ans;
}
```

要点：
- **存下标**便于判过期
- 维护**值递减**的队列 → 队首是最大值
- 求最小值就改成**值递增**

## 直觉：为什么是 O(N)

每个下标**入队一次、出队一次**——总操作 O(N)。

**关键洞察**：当 `nums[i]` 进来时，**所有比它小的旧元素永远不会再成为窗口最大值**——因为：
- 在 i 加入后的窗口里，`nums[i]` 始终更大
- 这些旧元素被 i 在窗口内"压"住了

所以可以安全地从队尾弹掉所有 `≤ nums[i]` 的元素。

## 何时想到单调队列

| 信号 | 用法 |
|---|---|
| "滑动窗口最大值 / 最小值" | 直接套 |
| "DP 转移要求"max/min over 一段窗口"" | 单调队列优化 DP |
| "最长 / 最短窗口 + 满足某最值约束" | 单调队列 + 滑窗 |
| "差分数组上的窗口最值" | 单调队列 |

## 经典题型

| 题 | 用法 |
|---|---|
| 滑动窗口最大值 LC 239 | 模板 |
| 绝对差不超过 K 的最长子数组 LC 1438 | 双单调队列（max + min） |
| 跳跃游戏 VI LC 1696 | DP + 单调队列优化 |
| 和至少为 K 的最短子数组 LC 862 | 前缀和 + 单调队列 |
| 队列的最大值（剑指 Offer 59-II） | 单调队列模拟 |
| 满足条件的子序列数目 | 排序 + 滑窗 |

### 例：DP 优化（跳跃游戏 VI）

朴素 DP：`dp[i] = max(dp[i-k..i-1]) + nums[i]`，O(NK)。

用单调队列维护 `dp` 在窗口 `[i-k, i-1]` 上的最大值，O(N)：

```cpp
int maxResult(vector<int>& nums, int k) {
    int n = nums.size();
    vector<int> dp(n);
    dp[0] = nums[0];
    deque<int> dq;
    dq.push_back(0);
    for (int i = 1; i < n; ++i) {
        if (dq.front() < i - k) dq.pop_front();         // 过期
        dp[i] = dp[dq.front()] + nums[i];               // 队首是窗口内最大 dp
        while (!dq.empty() && dp[dq.back()] <= dp[i]) dq.pop_back();
        dq.push_back(i);
    }
    return dp[n - 1];
}
```

这是**单调队列优化 DP** 的经典案例——把 O(NK) 降到 O(N)。

## 单调队列 vs 优先队列

| 维度 | 单调队列 | 优先队列（[[heap\|堆]]） |
|---|---|---|
| 滑窗最值 | O(N) | O(N log K) |
| 实现 | deque + 简单逻辑 | priority_queue + 懒删除 |
| 适用 | **窗口固定大小**或**单调推进** | 任意场景 |
| 不能做 | 任意位置删除 | — |

绝大多数滑窗最值题用单调队列，**比堆快、代码也短**。

## 易错点

- 用 `queue` 写 → 不能 pop_back；必须 `deque`
- 队列里存值还是下标？**存下标**便于过期判断
- 比较符号 `<=` vs `<` 决定相同值是否替换队尾元素：通常用 `<=` （挤掉旧的）
- 收集答案的时机：定长窗口 `i >= k - 1` 才开始
- 求最小值时把比较反过来：`nums[dq.back()] >= nums[i]` 时 pop

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「单调队列」相关关键词（单调队列）的笔记，按来源分组。

### 灵茶山艾府 + 代码随想录
- [[课程/灵茶山艾府和代码随想录/单调队列|课程/灵茶山艾府 + 代码随想录/单调队列.md]]

### 左程云
- [[课程/灵茶山艾府和代码随想录/单调队列|课程/左程云/002. 知识点总结/20. 算法知识点/单调队列.md]]

## 参考资料

- [[../../课程/灵茶山艾府 + 代码随想录/单调队列]] — 你的单调队列专题
- [[../../课程/左程云/100. 算法课程]] — 左程云单调队列章节

## 关联条目

- 上层：[[algorithm-overview]]
- 数据结构：[[queue]]
- 兄弟：[[monotonic-stack]] · [[heap]]
- 应用：[[sliding-window]] · [[dp]]
