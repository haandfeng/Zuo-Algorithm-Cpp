---
title: 接雨水
aliases: [Trapping Rain Water, LC 42]
tags: [problem, two-pointers, monotonic-stack, classic]
difficulty: hard
leetcode: 42
created: 2026-05-03
updated: 2026-05-03
---

# 接雨水（LC 42）

> **题号**：LeetCode 42
> **难度**：Hard
> **核心**：**3 种解法都要会**——动态规划 / [[../techniques/two-pointers|双指针]] / [[../techniques/monotonic-stack|单调栈]]。是面试**最最经典**的题之一。

## 题目

给定 N 个非负整数表示每个柱子的高度，计算下完雨能装多少水。

```text
height = [0,1,0,2,1,0,1,3,2,1,2,1]
水量   = 6
```

直观：每个位置的存水量 = `min(左侧最高, 右侧最高) - 自己高度`（若负则 0）。

## 解法 1：动态规划（左右最大值）

```cpp
int trap(vector<int>& h) {
    int n = h.size();
    if (n == 0) return 0;
    vector<int> lmax(n), rmax(n);
    lmax[0] = h[0];
    for (int i = 1; i < n; ++i) lmax[i] = max(lmax[i - 1], h[i]);
    rmax[n - 1] = h[n - 1];
    for (int i = n - 2; i >= 0; --i) rmax[i] = max(rmax[i + 1], h[i]);
    int ans = 0;
    for (int i = 0; i < n; ++i) ans += min(lmax[i], rmax[i]) - h[i];
    return ans;
}
```

时间 O(N)、空间 O(N)。**最直观**，但空间不是最优。

## 解法 2：双指针 O(1) 空间

把"左侧最高 / 右侧最高"用两个变量边走边维护：

```cpp
int trap(vector<int>& h) {
    int l = 0, r = h.size() - 1;
    int lmax = 0, rmax = 0, ans = 0;
    while (l < r) {
        if (h[l] < h[r]) {
            lmax = max(lmax, h[l]);
            ans += lmax - h[l];
            l++;
        } else {
            rmax = max(rmax, h[r]);
            ans += rmax - h[r];
            r--;
        }
    }
    return ans;
}
```

时间 O(N)、空间 O(1)。

**直觉**：当 `h[l] < h[r]` 时，**l 位置一定被 lmax 限制**——因为右边一定存在一个 ≥ rmax ≥ h[r] > h[l] 的柱子能挡住水，所以右侧的最大不会更小。这个性质是双指针正确的关键。

## 解法 3：单调栈

逐一入栈，遇到比栈顶高的就**结算栈顶能接的水**：

```cpp
int trap(vector<int>& h) {
    stack<int> st;            // 单调递减栈，存下标
    int ans = 0;
    for (int i = 0; i < (int)h.size(); ++i) {
        while (!st.empty() && h[i] > h[st.top()]) {
            int bottom = st.top(); st.pop();
            if (st.empty()) break;
            int left = st.top();
            int width = i - left - 1;
            int height = min(h[i], h[left]) - h[bottom];
            ans += width * height;
        }
        st.push(i);
    }
    return ans;
}
```

时间 O(N)、空间 O(N)。**是 [[../techniques/monotonic-stack|单调栈]] 的范例题之一**。

**直觉**：单调栈逐"层"结算——对一个被夹在中间的低柱子，遇到右边第一个比它高的（i）和它在栈里的左邻居（left），它们三个能形成一个长方形容器。

## 三种解法对比

| 解法 | 时间 | 空间 | 思维 |
|---|---|---|---|
| DP | O(N) | O(N) | 最直观 |
| 双指针 | O(N) | **O(1)** | 最巧妙 |
| 单调栈 | O(N) | O(N) | 模板化、可推广 |

面试**最佳答案**：先讲 DP 让面试官确认思路，再优化到双指针。如果问"还有别的解法吗"再补单调栈。

## 变体

| 题 | 关键差异 |
|---|---|
| 接雨水 II（二维）LC 407 | **小顶堆** 从边界向内推 |
| 柱状图最大矩形 LC 84 | 见 [[largest-rectangle]]，单调栈兄弟题 |
| 盛最多水的容器 LC 11 | 双指针，比这题简单（不要求里面存的水） |

### 接雨水 II（二维）

二维不能简单用左右最大——要从**边界**向**内部**用**小顶堆**：

```cpp
int trapRainWater(vector<vector<int>>& h) {
    int n = h.size(), m = h[0].size();
    vector<vector<bool>> vis(n, vector<bool>(m, false));
    priority_queue<tuple<int,int,int>, vector<tuple<int,int,int>>, greater<>> pq;
    // 边界入堆
    for (int i = 0; i < n; ++i) for (int j = 0; j < m; ++j) {
        if (i == 0 || i == n-1 || j == 0 || j == m-1) {
            pq.push({h[i][j], i, j});
            vis[i][j] = true;
        }
    }
    int ans = 0;
    int dx[] = {-1,1,0,0}, dy[] = {0,0,-1,1};
    while (!pq.empty()) {
        auto [cur, x, y] = pq.top(); pq.pop();
        for (int d = 0; d < 4; ++d) {
            int nx = x + dx[d], ny = y + dy[d];
            if (nx < 0 || nx >= n || ny < 0 || ny >= m || vis[nx][ny]) continue;
            ans += max(0, cur - h[nx][ny]);
            pq.push({max(cur, h[nx][ny]), nx, ny});
            vis[nx][ny] = true;
        }
    }
    return ans;
}
```

## 易错点

- 双指针的判断条件：**比较 `h[l]` 和 `h[r]`**，不是 `lmax` 和 `rmax`
- 单调栈结算时**别忘了 `st.empty()` 后 break**——没有左边界就不能形成容器
- `width = i - left - 1` 不要写成 `i - left`
- 二维版本把"高度" 当 `max(cur, h[nx][ny])` 推进——不要写 `cur`

## 在你 Vault 中的位置

- [[../../题单/Hot100|Hot100]] 必考
- [[../../题单/Blind75|Blind75]] 必考
- [[../../课程/灵茶山艾府和代码随想录/单调栈]] 中有详细讲解

## 关联条目

- 技巧：[[../techniques/two-pointers]] · [[../techniques/monotonic-stack]] · [[../techniques/prefix-sum]]
- 兄弟题：[[largest-rectangle]]
