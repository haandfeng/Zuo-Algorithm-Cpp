---
title: 贪心
aliases: [Greedy, 贪心算法]
tags: [pattern]
created: 2026-05-03
updated: 2026-05-03
---

# 贪心

> **TL;DR**：贪心 = **每一步都做局部最优选择，期望最终全局最优**。**不一定对**——贪心需要**证明**或**对照 DP 验证**。能贪通常很快（O(N) 或 O(N log N)）。

## 何时贪心是对的

形式化：

- **贪心选择性质**：局部最优能导致全局最优
- **最优子结构**：原问题的最优解包含子问题的最优解

实战中**没法严格证明就要谨慎**——很多看起来该贪的题其实需要 [[dp]]（如背包、LIS）。

## 三类常见贪心

### 1. 排序后扫描

按某关键字排序，然后线性扫一遍贪心选取。

```cpp
// LC 455 分发饼干
int findContentChildren(vector<int>& g, vector<int>& s) {
    sort(g.begin(), g.end());
    sort(s.begin(), s.end());
    int i = 0, j = 0, ans = 0;
    while (i < (int)g.size() && j < (int)s.size()) {
        if (s[j] >= g[i]) { ans++; i++; }
        j++;
    }
    return ans;
}
```

### 2. 贪心 + 优先队列

```cpp
// LC 1834 单线程 CPU 任务调度
vector<int> getOrder(vector<vector<int>>& tasks) {
    int n = tasks.size();
    vector<tuple<int,int,int>> arr;
    for (int i = 0; i < n; ++i) arr.push_back({tasks[i][0], tasks[i][1], i});
    sort(arr.begin(), arr.end());
    priority_queue<pair<int,int>, vector<pair<int,int>>, greater<>> pq;
    long long now = 0;
    int idx = 0;
    vector<int> ans;
    while (ans.size() < (size_t)n) {
        while (idx < n && get<0>(arr[idx]) <= now) {
            pq.push({get<1>(arr[idx]), get<2>(arr[idx])});
            idx++;
        }
        if (pq.empty()) { now = get<0>(arr[idx]); continue; }
        auto [d, id] = pq.top(); pq.pop();
        now += d;
        ans.push_back(id);
    }
    return ans;
}
```

### 3. 区间贪心

按起点 / 终点排序，然后线性合并 / 选择。

```cpp
// LC 56 合并区间
vector<vector<int>> merge(vector<vector<int>>& intervals) {
    sort(intervals.begin(), intervals.end());
    vector<vector<int>> ans;
    for (auto& iv : intervals) {
        if (!ans.empty() && iv[0] <= ans.back()[1])
            ans.back()[1] = max(ans.back()[1], iv[1]);
        else
            ans.push_back(iv);
    }
    return ans;
}
```

```cpp
// LC 435 无重叠区间（按结束时间排序贪心）
int eraseOverlapIntervals(vector<vector<int>>& intervals) {
    sort(intervals.begin(), intervals.end(),
         [](auto& a, auto& b) { return a[1] < b[1]; });
    int end = INT_MIN, cnt = 0;
    for (auto& iv : intervals) {
        if (iv[0] >= end) end = iv[1];        // 选这个
        else cnt++;                            // 重叠，删掉
    }
    return cnt;
}
```

按**结束时间**排序是这道题贪心正确的关键——经典证明用**交换论证**（exchange argument）。

## 贪心 vs DP

| 维度 | 贪心 | DP |
|---|---|---|
| 决策 | 一步定终身 | 维护多个状态 |
| 复杂度 | O(N) 或 O(N log N) | O(N²)、O(N · K) |
| 正确性 | 需证明（难） | 状态定义对就一定对 |
| 适用 | 有"贪心选择性质" | 大部分最优化问题 |

**不会证就先写 DP 兜底**，再看能不能简化为贪心。

经典反例：

- **0-1 背包**：贪心按"价值 / 重量"排序选 → 错。要 DP（[[dp-knapsack]]）
- **完全背包 + 找零**：硬币 [1, 5, 10, 25] 贪心对，但 [1, 3, 4] 凑 6 时贪心错（4+1+1=3 个，DP 给 3+3=2 个）

## 经典题型

| 题 | 贪心策略 |
|---|---|
| 分发饼干 LC 455 | 排序双指针 |
| 跳跃游戏 LC 55 | 维护当前最远 |
| 跳跃游戏 II LC 45 | 当前层 / 下一层最远 |
| 加油站 LC 134 | 累计差值 |
| 合并区间 LC 56 | 按起点排序 |
| 无重叠区间 LC 435 | 按终点排序 |
| 用最少箭射爆气球 LC 452 | 按右端点排序 |
| 划分字母区间 LC 763 | 记录每个字符最后位置 |
| 任务调度器 LC 621 | 大顶堆 + 冷却 |
| 重构字符串 LC 767 | 大顶堆 |
| 摆动序列 LC 376 | 计极值点数 |
| 买卖股票的最佳时机 II LC 122 | 涨就买卖 |
| K 次取反后最大化数组和 LC 1005 | 排序 + 贪心 |
| 柠檬水找零 LC 860 | 模拟 + 贪心 |
| 根据身高重建队列 LC 406 | 排序 + 插入 |
| 分发糖果 LC 135 | 双向扫 |
| 雇佣 K 名工人的最低成本 LC 857 | 排序 + 堆 |

## 证明三种思路

### 1. 交换论证（Exchange Argument）

假设最优解中相邻两个决策可以交换，证明交换不会变差 → 可以重排为贪心解。

适用：区间调度、排序贪心。

### 2. 归纳法

证明每一步贪心选择都"不会比最优解差"，归纳到结尾。

### 3. 反证法

假设贪心不是最优 → 推矛盾。

## 易错点

- **没证明就贪** → 反例容易跑出来
- 排序的关键字选错（按起点 vs 按终点 vs 按差值）
- 边界情况：空数组、单元素、所有相同值
- 贪心后还要验证答案是否真的可行
- 贪心需要的数据结构（堆、deque）忘了维护

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「贪心」相关关键词（贪心）的笔记，按来源分组。

### 灵茶山艾府和代码随想录
- [[贪心|课程/灵茶山艾府和代码随想录/贪心.md]]

### 左程云
- [[_贪心算法Memo|课程/左程云/001. 难点重点照顾/_贪心算法Memo.md]]
- [[在动态规划中, 贪心是个什么地位|课程/左程云/001. 难点重点照顾/在动态规划中, 贪心是个什么地位.md]]
- [[贪心算法|课程/左程云/002. 知识点总结/20. 算法知识点/贪心算法.md]]
- [[13 贪心算法（上）|课程/左程云/100. 算法课程/01. 体系学习班/13 贪心算法（上）.md]]
- [[14 贪心算法（下）|课程/左程云/100. 算法课程/01. 体系学习班/14 贪心算法（下）.md]]
- [[09 贪心算法|课程/左程云/100. 算法课程/10. 基础课/09 贪心算法.md]]

## 参考资料

- [[../../课程/灵茶山艾府和代码随想录/贪心]] — 你的贪心专题
- 《算法导论》第 16 章

## 关联条目

- 上层：[[algorithm-overview]]
- 兄弟：[[dp]]
- 数据结构：[[heap]]
- 应用：[[mst]]（Kruskal 是贪心）· [[shortest-path]]（Dijkstra 是贪心）
