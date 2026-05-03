---
title: 离散化
aliases: [Discretization, 坐标压缩]
tags: [technique]
created: 2026-05-03
updated: 2026-05-03
---

# 离散化

> **TL;DR**：当数据**值域很大但不同值的个数少**时，把每个值映射成它在排序后的下标。常配合 [[../data-structures/fenwick-tree|树状数组]] / [[../data-structures/segment-tree|线段树]] 使用。

## 问题

值域 ∈ [-1e9, 1e9]，但只有 1e5 个不同值。直接开数组 → 内存爆。
解决方案：**按出现的值排序去重 → 每个值用它的排名（下标）替代**。

## 模板

```cpp
vector<int> values = nums;
sort(values.begin(), values.end());
values.erase(unique(values.begin(), values.end()), values.end());

auto rk = [&](int x) {
    return lower_bound(values.begin(), values.end(), x) - values.begin() + 1;   // 1-indexed
};

// 之后所有用 a[i] 的地方改成 rk(a[i])
```

时间 O(N log N) 预处理，单次查询 O(log N)。

## 何时用

| 信号 | 例子 |
|---|---|
| "数据范围 1e9 但 N ≤ 1e5" | 离散化 |
| "求逆序对 / 区间小于某值的数" | 离散化 + BIT |
| "区间染色 / 段合并" + 大值域 | 离散化 + 区间数据结构 |

## 经典题型

| 题 | 用法 |
|---|---|
| 求逆序对 | 离散化 + BIT，见 [[../data-structures/fenwick-tree]] |
| 计算右侧小于当前元素的个数 LC 315 | 离散化 + BIT |
| 区间和的个数 LC 327 | 离散化 + BIT / 归并 |
| 翻转对 LC 493 | 离散化 + BIT |
| 矩形面积 II LC 850 | 离散化 + 扫描线（[[scan-line]]） |
| 我的日程安排表 II LC 731 | 离散化 + 差分 |

## 易错点

- **`+ 1`**：用 1-indexed 配合 BIT 时不要忘
- 浮点数离散化要小心精度
- 离散化后**记得用排名查询**——直接用原值会算错
- `unique` 之前必须 `sort`

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「离散化」相关关键词（离散化）的笔记，按来源分组。

### 左程云
- [[数据离散化|课程/左程云/004. 刷题经验总结/数据离散化.md]]

## 关联条目

- 上层：[[../maps/algorithm-overview]]
- 配合：[[../data-structures/fenwick-tree]] · [[../data-structures/segment-tree]]
- 兄弟：[[scan-line]]
