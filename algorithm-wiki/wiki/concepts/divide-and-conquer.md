---
title: 分治
aliases: [Divide and Conquer, 分而治之]
tags: [concept, paradigm]
created: 2026-05-03
updated: 2026-05-03
---

# 分治

> **TL;DR**：分治 = **分（divide）** 把大问题切成同形小问题 + **治（conquer）** 递归解决小问题 + **合（combine）** 把小问题答案拼成大问题答案。是 [[recursion|递归]] 的一种典型用法。

## 三段式骨架

```cpp
ResultType solve(Problem p) {
    if (p 是最小问题) return base_answer(p);    // 治到底了
    auto [p1, p2, ...] = divide(p);             // 分
    auto r1 = solve(p1);                         // 治
    auto r2 = solve(p2);
    return combine(r1, r2, ...);                 // 合
}
```

不同算法在三段上各有重头：

| 算法 | divide 重 | combine 重 |
|---|---|---|
| 二分查找 | ✅ | — |
| 快速排序 | ✅ | — |
| 归并排序 | — | ✅ |
| 大整数乘法 / Karatsuba | ✅ | ✅ |
| 最近点对 | ✅ | ✅ |

## 经典分治算法

### 1. 归并排序

```cpp
void mergeSort(vector<int>& a, int l, int r) {
    if (l >= r) return;                    // base
    int mid = l + (r - l) / 2;
    mergeSort(a, l, mid);                  // 治左
    mergeSort(a, mid + 1, r);              // 治右
    merge(a, l, mid, r);                   // 合
}
```

复杂度 $T(N) = 2T(N/2) + O(N) = O(N \log N)$。详见 [[sorting]]。

### 2. 快速排序

```cpp
void quickSort(vector<int>& a, int l, int r) {
    if (l >= r) return;
    int pivot = partition(a, l, r);        // 分（核心）
    quickSort(a, l, pivot - 1);            // 治
    quickSort(a, pivot + 1, r);
    // combine 不做（in-place）
}
```

平均 $T(N) = 2T(N/2) + O(N) = O(N \log N)$，最坏 $O(N^2)$。

### 3. 二分查找

```cpp
int binarySearch(vector<int>& a, int target) {
    int l = 0, r = a.size() - 1;
    while (l <= r) {
        int m = l + (r - l) / 2;
        if (a[m] == target) return m;
        else if (a[m] < target) l = m + 1;     // 弃左半
        else                    r = m - 1;     // 弃右半
    }
    return -1;
}
```

每次砍一半，$T(N) = T(N/2) + O(1) = O(\log N)$。详见 [[binary-search]]。

### 4. 求逆序对（归并副产品）

在归并的合并步骤里顺手统计左半元素 > 右半元素的对数。复杂度同归并 $O(N \log N)$。

```cpp
long long mergeCount(vector<int>& a, int l, int r) {
    if (l >= r) return 0;
    int mid = l + (r - l) / 2;
    long long cnt = mergeCount(a, l, mid) + mergeCount(a, mid + 1, r);
    // 合并 + 数逆序
    vector<int> tmp;
    int i = l, j = mid + 1;
    while (i <= mid && j <= r) {
        if (a[i] <= a[j]) tmp.push_back(a[i++]);
        else { tmp.push_back(a[j++]); cnt += mid - i + 1; }   // a[i..mid] 都 > a[j]
    }
    while (i <= mid) tmp.push_back(a[i++]);
    while (j <= r)   tmp.push_back(a[j++]);
    copy(tmp.begin(), tmp.end(), a.begin() + l);
    return cnt;
}
```

### 5. 最大子数组和（分治版）

虽然 Kadane 算法 O(N) 更优，但分治版能上 $T(N) = 2T(N/2) + O(N) = O(N \log N)$，是经典分治练习题。

## 复杂度分析（Master 定理）

分治的复杂度通常用 [[complexity#Master 定理]] 套：

$$
T(N) = a \cdot T(N/b) + f(N)
$$

| 算法 | a | b | f(N) | T(N) |
|---|---|---|---|---|
| 二分 | 1 | 2 | O(1) | O(log N) |
| 归并 | 2 | 2 | O(N) | O(N log N) |
| Strassen 矩阵乘 | 7 | 2 | O(N²) | O(N^2.81) |
| 快速选择（Quickselect） | 1 | 2（平均） | O(N) | O(N) |

## 何时想到分治

信号词：

- "求第 K 大 / 第 K 小"（[[problems/lis|和这个无关，举例]] / Quickselect）
- "在有序数组里找…"（[[binary-search]]）
- "求逆序对 / 计数对"（归并副产品）
- "二维平面最近点对"
- "区间合并 / 区间统计"（也常用 [[segment-tree|线段树]]）

## 分治 vs 动态规划

两者都是"用小问题解大问题"，区别：

| 维度 | 分治 | DP |
|---|---|---|
| 子问题独立 | ✅ | 通常**重叠** |
| 是否记忆 | 不需要 | 必须（否则指数爆炸） |
| 自顶向下还是自底向上 | 自顶向下 | 都行 |
| 经典示例 | 归并、快排、二分 | 背包、LIS、编辑距离 |

口诀：**子问题不重叠 → 分治；重叠 → DP**。

## 常见 BUG

1. **base case 写错**（差 1）→ 死循环或漏算
2. **divide 不平衡**：例如 `mid = (l + r) / 2` 在 `l = r` 时不会缩小 → 死循环
3. **combine 写错**：例如归并的合并指针越界
4. **栈溢出**：分治深度 = $O(\log N)$ 通常没事，但极端不平衡（最坏快排）会到 O(N)

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「分治」相关关键词（分治）的笔记，按来源分组。

### 左程云
- [[分治|课程/左程云/002. 知识点总结/20. 算法知识点/分治.md]]

## 参考资料

- [[../../课程/C++算法/必备/快排]] — 你写的快排笔记
- [[../../课程/C++算法/必备/归并]] — 你写的归并笔记
- 《算法导论》第 4 章

## 关联条目

- 上层：[[algorithm-overview]]
- 工具：[[recursion]]
- 应用：[[sorting]] · [[binary-search]]
- 对比：[[dp]]
