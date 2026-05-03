---
title: 扫描线
aliases: [Scan Line, Sweep Line, 区间合并]
tags: [technique]
created: 2026-05-03
updated: 2026-05-03
---

# 扫描线

> **TL;DR**：把"区间 / 矩形"事件按一个维度（通常是 x 坐标）排序，**用一条假想的线从左往右扫**，扫过事件时增删活跃集合。**区间合并、矩形面积、活动调度**的通用解法。

## 1D 扫描线：区间合并

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

本质：每个区间产生 `+1`（开始）和 `-1`（结束）两个事件，按位置排序。

## 1D 扫描线：差分版

按"事件 +1 / -1" 累加，方便统计某点的活跃数：

```cpp
// 同时多少会议进行
vector<pair<int,int>> events;       // {time, +1 / -1}
for (auto& [s, e] : meetings) {
    events.push_back({s, +1});
    events.push_back({e, -1});
}
sort(events.begin(), events.end());
int active = 0, peak = 0;
for (auto& [t, delta] : events) {
    active += delta;
    peak = max(peak, active);
}
return peak;
```

LC 253 会议室 II 的另一种解法。

## 2D 扫描线：矩形面积并

求多个矩形的并集面积。

**思路**：用 x 坐标作为扫描方向。每个矩形产生**左、右**两个事件。维护"当前活跃 y 区间总长度"——用 [[../data-structures/segment-tree|线段树]]。

```cpp
struct Event { int x, y1, y2, type; };       // type +1 / -1

long long rectangleArea(vector<vector<int>>& rects) {
    vector<Event> events;
    vector<int> ys;
    for (auto& r : rects) {
        events.push_back({r[0], r[1], r[3], +1});
        events.push_back({r[2], r[1], r[3], -1});
        ys.push_back(r[1]);
        ys.push_back(r[3]);
    }
    sort(events.begin(), events.end(),
         [](auto& a, auto& b) { return a.x < b.x; });
    sort(ys.begin(), ys.end());
    ys.erase(unique(ys.begin(), ys.end()), ys.end());

    // ... 需要维护 y 上的"被多少矩形覆盖"，用线段树（懒标记）
    // 略：完整实现见 LC 850
    return 0;
}
```

LC 850 矩形面积 II 是经典扫描线 + 线段树题。

## 何时想到扫描线

| 信号 | 用法 |
|---|---|
| "区间合并 / 区间是否重叠" | 1D 扫描线 |
| "同时进行的活动 / 任务数" | 差分扫描 |
| "矩形面积 / 周长并" | 2D 扫描 + 线段树 |
| "区间染色后查总染色长度" | 扫描 + 数据结构 |

## 经典题型

| 题 | 类型 |
|---|---|
| 合并区间 LC 56 | 1D 扫描 |
| 插入区间 LC 57 | 1D 扫描 |
| 会议室 LC 252 / 253 | 1D 扫描 / 差分 |
| 用最少箭射爆气球 LC 452 | 1D 扫描 |
| 我的日程安排表系列 LC 729/731/732 | 扫描 / map 差分 |
| 矩形面积 LC 223 | 计算几何（不算扫描线） |
| 矩形面积 II LC 850 | 2D 扫描 + 线段树 |
| 天际线问题 LC 218 | 扫描线 + 优先队列 |

## 易错点

- 排序时**起点相同**怎么处理：通常 `+1` 优先 `-1`（防止瞬时区间被漏）
- 端点是开区间还是闭区间：`[s, e]` vs `[s, e)`
- 离散化后 y 坐标 → 线段树下标
- 矩形面积**累加**要乘 (x_next - x_curr)

## 关联条目

- 上层：[[../maps/algorithm-overview]]
- 数据结构：[[../data-structures/segment-tree]]
- 兄弟：[[discretization]] · [[prefix-sum]]
