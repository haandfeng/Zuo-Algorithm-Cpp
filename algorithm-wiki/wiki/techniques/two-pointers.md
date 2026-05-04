---
title: 双指针
aliases: [Two Pointers, 快慢指针, 对撞指针]
tags: [technique]
created: 2026-05-03
updated: 2026-05-03
---

# 双指针

> **TL;DR**：用**两个指针**协同扫描数据，把 O(N²) 暴力降到 O(N)。三种常见模式：**对撞双指针**、**同向双指针（即 [[sliding-window|滑动窗口]]）**、**快慢双指针**。

## 三种模式

### 1. 对撞（相向）双指针

两个指针从两端往中间走。**前提**：序列**有序**或具备某种"两端选择"的对称性。

```cpp
int l = 0, r = n - 1;
while (l < r) {
    if (满足) l++;
    else      r--;
}
```

经典代表：

- 两数之和 II（有序数组）LC 167
- 三数之和 LC 15（排序后内层用对撞）
- 接雨水 LC 42（两端往中间扫）
- 反转字符串 LC 344
- 验证回文 LC 125

### 2. 同向双指针（[[sliding-window|滑动窗口]]）

两个指针都从左往右，**右指针扩展**、**左指针收缩**：

```cpp
int l = 0;
for (int r = 0; r < n; ++r) {
    // 加入 a[r]
    while (窗口不合法) {
        // 移出 a[l]
        l++;
    }
    // 此时 [l, r] 是合法窗口，更新答案
}
```

详见 [[sliding-window]]。代表题：

- 无重复字符的最长子串 LC 3
- 长度最小的子数组 LC 209
- 最小覆盖子串 LC 76

### 3. 快慢指针

两个指针**步速不同**。最经典：

**链表找中点**：

```cpp
ListNode *slow = head, *fast = head;
while (fast && fast->next) {
    slow = slow->next;
    fast = fast->next->next;
}
return slow;
```

**链表判环**（Floyd 判圈）：

```cpp
while (fast && fast->next) {
    slow = slow->next;
    fast = fast->next->next;
    if (slow == fast) return true;
}
return false;
```

**数组去重**（LC 26）：

```cpp
int slow = 0;
for (int fast = 0; fast < n; ++fast) {
    if (a[fast] != a[slow]) a[++slow] = a[fast];
}
return slow + 1;
```

## 何时想到双指针

| 信号 | 模式 |
|---|---|
| "有序数组 + 两数和 / 三数和" | 对撞 |
| "回文判定" | 对撞 |
| "数组中找两个数满足某关系" | 对撞（先排序） |
| "连续子数组 / 子串 + 满足某约束" | 同向（滑窗） |
| "最长 / 最短满足条件的子段" | 同向 |
| "去除重复 / 原地操作数组" | 快慢 |
| "链表找中点 / 判环" | 快慢 |
| "倒数第 K 个节点" | 快慢（先走 K 步再同步） |

## 经典题型（对撞）

```cpp
// 三数之和 LC 15
vector<vector<int>> threeSum(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    vector<vector<int>> res;
    int n = nums.size();
    for (int i = 0; i < n - 2; ++i) {
        if (i > 0 && nums[i] == nums[i - 1]) continue;     // 去重
        int l = i + 1, r = n - 1;
        while (l < r) {
            int s = nums[i] + nums[l] + nums[r];
            if (s == 0) {
                res.push_back({nums[i], nums[l], nums[r]});
                while (l < r && nums[l] == nums[l + 1]) l++;
                while (l < r && nums[r] == nums[r - 1]) r--;
                l++; r--;
            } else if (s < 0) l++;
            else              r--;
        }
    }
    return res;
}
```

```cpp
// 盛最多水的容器 LC 11
int maxArea(vector<int>& h) {
    int l = 0, r = h.size() - 1, ans = 0;
    while (l < r) {
        ans = max(ans, min(h[l], h[r]) * (r - l));
        if (h[l] < h[r]) l++;
        else             r--;
    }
    return ans;
}
```

```cpp
// 接雨水 LC 42（双指针版）
int trap(vector<int>& h) {
    int l = 0, r = h.size() - 1, lmax = 0, rmax = 0, ans = 0;
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

详见 [[problems/trapping-rain-water]]。

## 复杂度

绝大多数双指针题：**时间 O(N)，空间 O(1)**。每个元素被两个指针各访问 ≤ 1 次（摊还）。

## 易错点

- 排序后**忘了去重**——例如三数之和漏去重会有重复答案
- 对撞指针的循环条件 `l < r` 还是 `l <= r`：求两个不同位置用 `l < r`
- 快慢指针的"找中点"：偶数长度想要前 / 后哪个？模板有差别
- 滑窗的 while 收缩条件**应该是"窗口不合法"**而不是"达到目标"
- 链表的快慢指针检查**fast 和 fast->next 都要非空**

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「双指针」相关关键词（双指针）的笔记，按来源分组。

### 灵茶山艾府和代码随想录
- [[双向双指针|课程/灵茶山艾府和代码随想录/双向双指针.md]]
- [[滑动窗口(同向双指针)|课程/灵茶山艾府和代码随想录/滑动窗口(同向双指针).md]]

### 左程云
- [[双指针-首尾指针法|课程/左程云/003. 编程基础技巧及工具/双指针-首尾指针法.md]]

### labuladong
- [[双指针-数组|课程/labuladong/核心刷题框架/双指针-数组.md]]
- [[双指针-链表 和 链表代码实现|课程/labuladong/核心刷题框架/双指针-链表 和 链表代码实现.md]]

## 参考资料

- [[../../课程/灵茶山艾府和代码随想录/双向双指针]] — 对撞双指针专题
- [[../../课程/灵茶山艾府和代码随想录/滑动窗口(同向双指针)]] — 同向双指针专题
- [[../../课程/labuladong/核心刷题框架]] — labuladong 双指针框架

## 关联条目

- 上层：[[algorithm-overview]]
- 衍生：[[sliding-window]]
- 数据结构：[[array]] · [[linked-list]]
- 应用：[[problems/trapping-rain-water]] · [[problems/lis]]
