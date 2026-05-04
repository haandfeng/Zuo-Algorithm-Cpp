---
title: 二分查找
aliases: [Binary Search, 折半查找]
tags: [algorithm, search]
created: 2026-05-03
updated: 2026-05-03
---

# 二分查找

> **TL;DR**：在**有序**或**有单调性的可判定空间**上二分。每次砍一半，O(log N)。**关键不是"会不会写循环"，而是"能不能想到二分"和"边界写不写错"**。

## 朴素：找等于 target

```cpp
int binarySearch(vector<int>& a, int target) {
    int l = 0, r = a.size() - 1;
    while (l <= r) {
        int m = l + (r - l) / 2;       // 防溢出
        if (a[m] == target) return m;
        else if (a[m] < target) l = m + 1;
        else                    r = m - 1;
    }
    return -1;
}
```

注意 `l + (r - l) / 2` 而不是 `(l + r) / 2` —— 后者在极大值时会**整数溢出**。

## 三种"找边界"模板

`std::lower_bound` 和 `std::upper_bound` 是面试和工程的"瑞士军刀"。

### 找第一个 `≥ target` 的下标（lower_bound）

```cpp
int lowerBound(vector<int>& a, int target) {
    int l = 0, r = a.size();             // 注意 r = size，半开区间
    while (l < r) {
        int m = l + (r - l) / 2;
        if (a[m] >= target) r = m;       // m 可能就是答案
        else                l = m + 1;
    }
    return l;                             // 可能 == size，表示找不到
}
```

### 找第一个 `> target` 的下标（upper_bound）

```cpp
int upperBound(vector<int>& a, int target) {
    int l = 0, r = a.size();
    while (l < r) {
        int m = l + (r - l) / 2;
        if (a[m] > target) r = m;
        else               l = m + 1;
    }
    return l;
}
```

### 找最后一个 `≤ target` 的下标

```cpp
int last = upperBound(a, target) - 1;
```

### 元素出现次数

```cpp
int count = upperBound(a, target) - lowerBound(a, target);
```

## C++ STL 直接调

```cpp
auto it = lower_bound(a.begin(), a.end(), target);
int idx = it - a.begin();             // 第一个 ≥ target 的下标
```

`std::set::lower_bound` 也存在但要 `s.lower_bound(x)`（成员函数）而不是 `lower_bound(s.begin(), s.end(), x)`，后者是 O(N) 因为 set 的迭代器是双向不是随机访问。

## 浮点数二分

精度模板：

```cpp
double l = 0, r = 1e9;
while (r - l > 1e-9) {                 // 注意精度，避免 ==
    double m = (l + r) / 2;
    if (check(m)) r = m;
    else          l = m;
}
return l;
```

或者**固定迭代次数**（更稳）：

```cpp
for (int iter = 0; iter < 100; ++iter) {
    double m = (l + r) / 2;
    if (check(m)) r = m;
    else          l = m;
}
```

100 次迭代精度大约 $2^{-100} \approx 10^{-30}$，远超任何题目需求。

## 二分的本质：单调可判定

**二分不只是"找数字"**——它适用于任何"答案在某个范围内、有 yes/no 单调性"的场景：

```text
答案空间：  [lo, hi]
              ↓ 单调性
判定函数：  check(mid) ∈ {true, false}
              ↓ 二分
返回最大 / 最小满足 check 的值
```

这就是 **[[binary-search-answer|二分答案]]**——非常重要的技巧，详见专门一文。

## 经典变体

### 1. 旋转排序数组找最小值（LC 153）

```cpp
int findMin(vector<int>& a) {
    int l = 0, r = a.size() - 1;
    while (l < r) {
        int m = l + (r - l) / 2;
        if (a[m] > a[r]) l = m + 1;
        else             r = m;
    }
    return a[l];
}
```

### 2. 旋转排序数组找 target（LC 33）

二分时判断 `[l, m]` 还是 `[m, r]` **哪一半是有序的**，再判断 target 是不是在其中。

### 3. 山脉数组峰值（LC 162）

```cpp
int peakIndex(vector<int>& a) {
    int l = 0, r = a.size() - 1;
    while (l < r) {
        int m = l + (r - l) / 2;
        if (a[m] < a[m + 1]) l = m + 1;
        else                 r = m;
    }
    return l;
}
```

### 4. 矩阵中的二分（LC 74 / 240）

行优先有序的矩阵：把二维下标拍扁成 `m = idx / cols`、`n = idx % cols`，按一维数组二分。

LC 240 行列双有序但不是全序：从右上角开始走（不是二分，但常一起讲）。

## 经典题型

| 题 | 类型 |
|---|---|
| 二分查找 LC 704 | 朴素 |
| 搜索插入位置 LC 35 | lower_bound |
| 在排序数组中查找元素第一/最后位置 LC 34 | lower + upper |
| 搜索旋转排序数组 LC 33 | 变体 |
| 寻找峰值 LC 162 | 变体 |
| Koko 吃香蕉 LC 875 | [[binary-search-answer]] |
| 分割数组的最大值 LC 410 | 二分答案 |
| H 指数 II LC 275 | 变体 |
| 寻找两个正序数组的中位数 LC 4 | 二分两个数组 |
| 长度最小子数组 LC 209 | 滑窗 + 可二分答案 |

## 易错点

| 错误 | 怎么避免 |
|---|---|
| 死循环 | 确保**每次循环 l/r 都改变**：`m+1` 或 `m-1` |
| `(l+r)/2` 溢出 | 用 `l + (r-l)/2` |
| 边界写错（差 1） | **写测试**：长度 1、2 的数组、target 在两端、不存在 |
| 区间开闭混淆 | **统一**用 `[l, r]` 或 `[l, r)`，别混用 |
| 模板背错 | 记**3 个模板**（找数 / lower / upper），别只记一个 |

**调试口诀**（labuladong）：
- "搜索区间是 `[l, r)`" → `r = nums.size()`，`while (l < r)`，`r = m` 或 `l = m + 1`
- "搜索区间是 `[l, r]`" → `r = nums.size() - 1`，`while (l <= r)`，`r = m - 1` 或 `l = m + 1`

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「二分查找」相关关键词（二分查找, 二分）的笔记，按来源分组。

### 灵茶山艾府和代码随想录
- [[二分查找|课程/灵茶山艾府和代码随想录/二分查找.md]]

### 左程云
- [[二分法|课程/左程云/002. 知识点总结/20. 算法知识点/二分法.md]]
- [[不是必须有序才能二分|课程/左程云/004. 刷题经验总结/不是必须有序才能二分.md]]
- [[二分法模板|课程/左程云/010. 代码模板/二分法模板.md]]
- [[03 二分、复杂度、动态数组、哈希表和有序表|课程/左程云/100. 算法课程/00. 新手班/03 二分、复杂度、动态数组、哈希表和有序表.md]]
- [[01 认识复杂度、对数器、二分法|课程/左程云/100. 算法课程/01. 体系学习班/01 认识复杂度、对数器、二分法.md]]
- [[01 认识复杂度、对数器、二分法与异或运算|课程/左程云/100. 算法课程/10. 基础课/01 认识复杂度、对数器、二分法与异或运算.md]]

### 题目
- [[0704-二分查找|题目/0704-二分查找.md]]

## 参考资料

- [[../../课程/灵茶山艾府和代码随想录/二分查找]] — 你的二分专题
- [[../../课程/labuladong/核心刷题框架]] — labuladong 二分模板
- [[../../课程/左程云/010. 代码模板]] — 左程云模板

## 关联条目

- 上层：[[algorithm-overview]]
- 衍生：[[binary-search-answer]]
- 应用：[[problems/lis]]（耐心排序版本用二分）
- 思维：[[divide-and-conquer]]
