---
title: 柱状图最大矩形
aliases: [Largest Rectangle in Histogram, LC 84]
tags: [problem, monotonic-stack, classic]
difficulty: hard
leetcode: 84
created: 2026-05-03
updated: 2026-05-03
---

# 柱状图最大矩形（LC 84）

> **题号**：LeetCode 84
> **难度**：Hard
> **核心**：[[../techniques/monotonic-stack|单调栈]] 的**镇山之宝**。一旦理解，[[trapping-rain-water|接雨水]]、LC 85 最大矩形、LC 1130 都顺。

## 题目

给定 N 根柱子的高度，求**用这些柱子能构成的最大矩形面积**——矩形必须**贴着 X 轴**、宽度是连续柱子。

```text
heights = [2,1,5,6,2,3]
最大矩形 = 10（高=5、宽=2，由下标 2~3）
```

## 朴素思路 O(N²)

枚举每根柱子作为**高度的瓶颈**，往左右扩到第一个比它矮的柱子，宽 = 距离。每根柱子 O(N)，总 O(N²)。

## 单调栈 O(N)

把"找左右第一个更小的柱子"这一步用 [[../techniques/monotonic-stack|单调栈]] 做到 O(1) 摊还，整体 O(N)。

```cpp
int largestRectangleArea(vector<int>& h) {
    int n = h.size();
    h.push_back(0);            // 末尾哨兵：迫使所有元素出栈结算
    stack<int> st;             // 存下标，对应高度递增
    int ans = 0;
    for (int i = 0; i <= n; ++i) {
        while (!st.empty() && h[i] < h[st.top()]) {
            int height = h[st.top()]; st.pop();
            int left = st.empty() ? -1 : st.top();
            int width = i - left - 1;
            ans = max(ans, height * width);
        }
        st.push(i);
    }
    h.pop_back();
    return ans;
}
```

时间 O(N)、空间 O(N)。

## 直觉

栈中的下标对应的高度**单调递增**。当遇到**更矮**的 `h[i]` 时：

- 栈顶的高度被"卡死"——**它的右边界就是 i**
- 它的**左边界** = 栈顶之前的栈顶（如果栈空就是 -1）
- 用**栈顶的高度作为矩形高**、左右边界算宽

末尾添加哨兵 0 是为了让所有未结算的元素都被强制 pop。

## 一图说明

```text
heights = [2, 1, 5, 6, 2, 3]
索引     =  0  1  2  3  4  5

i=0, push 0      栈: [0]
i=1, h[1]=1 < h[0]=2, 结算 0:
                 height=2, left=-1, width=1-(-1)-1=1, area=2
                 push 1                   栈: [1]
i=2, push 2      栈: [1, 2]
i=3, push 3      栈: [1, 2, 3]
i=4, h[4]=2 < h[3]=6, 结算 3:
                 height=6, left=2, width=4-2-1=1, area=6
                 h[4]=2 < h[2]=5, 结算 2:
                 height=5, left=1, width=4-1-1=2, area=10 ←
                 push 4                   栈: [1, 4]
i=5, push 5      栈: [1, 4, 5]
i=6 (哨兵), 全部结算...
```

## 变体题

### LC 85 最大矩形（二维）

把每一行当作一个"地基"：累加 1 的高度，再对每一行跑 LC 84。

```cpp
int maximalRectangle(vector<vector<char>>& mat) {
    if (mat.empty()) return 0;
    int n = mat.size(), m = mat[0].size();
    vector<int> heights(m, 0);
    int ans = 0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            heights[j] = mat[i][j] == '1' ? heights[j] + 1 : 0;
        }
        ans = max(ans, largestRectangleArea(heights));
    }
    return ans;
}
```

时间 O(NM)。

### LC 1856 子数组最小乘积的最大值

每个元素作为最小值的"贡献区间"用单调栈算，然后乘以这个区间的和（[[../techniques/prefix-sum|前缀和]]）。

### LC 907 子数组的最小值之和

类似：每个元素作为最小值的贡献区间数 × 它的值，求和。

## 易错点

- **没加哨兵 0** → 末尾的递增段不会被结算
- 比较符号 `<` vs `≤`：用 `<` 时相同高度不被结算（保留更左的），结果可能偏小；写法各有取舍但要一致
- `left = st.empty() ? -1 : st.top()` 别忘了空栈情况
- **矩形高用栈顶高度**而不是 i 位置高度
- 二维版本时 `heights[j] += 1` 错——要用 `mat[i][j] == '1' ? heights[j] + 1 : 0`

## 在你 Vault 中的位置

- [[../../Hot100|Hot100]] 必考
- [[../../零茶山艾府+代码随想录/单调栈]] 详细讲解

## 关联条目

- 技巧：[[../techniques/monotonic-stack]]
- 兄弟题：[[trapping-rain-water]]（同样基于"找左右第一个不同"）
- 数据结构：[[../data-structures/stack]]
