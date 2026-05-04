---
title: 排序
aliases: [Sorting, 十大排序, 快排, 归并, 堆排序]
tags: [algorithm, sorting]
created: 2026-05-03
updated: 2026-05-03
---

# 排序

> **TL;DR**：排序是算法学习的"开胃菜"。10 种经典排序按"复杂度 + 稳定性 + 是否原地"分类。**面试只需精通**：快排、归并、堆排序；**了解**：插入、计数、基数。

## 速查表

| 算法 | 平均 | 最坏 | 空间 | 稳定 | 原地 | 分类 |
|---|---|---|---|---|---|---|
| 冒泡 | O(N²) | O(N²) | O(1) | ✅ | ✅ | 比较 |
| 选择 | O(N²) | O(N²) | O(1) | ❌ | ✅ | 比较 |
| 插入 | O(N²) | O(N²) | O(1) | ✅ | ✅ | 比较 |
| 希尔 | O(N^1.3) | O(N²) | O(1) | ❌ | ✅ | 比较 |
| **归并** | **O(N log N)** | **O(N log N)** | O(N) | ✅ | ❌ | 比较 |
| **快排** | **O(N log N)** | O(N²) | O(log N) | ❌ | ✅ | 比较 |
| **堆排** | **O(N log N)** | **O(N log N)** | O(1) | ❌ | ✅ | 比较 |
| 计数 | O(N + K) | O(N + K) | O(K) | ✅ | ❌ | 非比较 |
| 基数 | O(N · L) | O(N · L) | O(N + K) | ✅ | ❌ | 非比较 |
| 桶 | O(N + K) | O(N²) | O(N + K) | ✅ | ❌ | 非比较 |

**N**：元素数；**K**：值域大小；**L**：位数。

**比较排序的下界是 O(N log N)**——任何只通过比较来排序的算法都不可能更快（信息论证明）。要更快只能利用值的额外信息（值域、位数），这就是非比较排序。

## 必须背熟的 3 种

### 1. 快速排序

```cpp
void quickSort(vector<int>& a, int l, int r) {
    if (l >= r) return;
    int pivot = a[l + (r - l) / 2];
    int i = l, j = r;
    while (i <= j) {
        while (a[i] < pivot) ++i;
        while (a[j] > pivot) --j;
        if (i <= j) {
            swap(a[i], a[j]);
            ++i; --j;
        }
    }
    quickSort(a, l, j);
    quickSort(a, i, r);
}
```

要点：
- **pivot 选择**影响最坏复杂度。固定 `a[l]` 在已排序数据上 O(N²)；选**中间**或**随机**或**三数取中**更安全
- **分区**有多种写法（Hoare、Lomuto、双指针填坑），选自己写得对的那种
- **快排不稳定**：相等元素可能被换到对方位置

### 2. 归并排序

```cpp
void mergeSort(vector<int>& a, int l, int r) {
    if (l >= r) return;
    int mid = l + (r - l) / 2;
    mergeSort(a, l, mid);
    mergeSort(a, mid + 1, r);
    // merge
    vector<int> tmp;
    int i = l, j = mid + 1;
    while (i <= mid && j <= r) {
        if (a[i] <= a[j]) tmp.push_back(a[i++]);
        else              tmp.push_back(a[j++]);
    }
    while (i <= mid) tmp.push_back(a[i++]);
    while (j <= r)   tmp.push_back(a[j++]);
    for (int k = 0; k < (int)tmp.size(); ++k) a[l + k] = tmp[k];
}
```

要点：
- **稳定**：相等元素保持原顺序（合并时左边优先）
- **空间 O(N)**：要开 tmp 数组
- 是 [[divide-and-conquer|分治]] 的范式
- **副产品**：可以顺手算[[concepts/divide-and-conquer#4. 求逆序对（归并副产品）|逆序对]]

### 3. 堆排序

```cpp
void heapify(vector<int>& a, int i, int n) {
    while (true) {
        int l = 2*i+1, r = 2*i+2, largest = i;
        if (l < n && a[l] > a[largest]) largest = l;
        if (r < n && a[r] > a[largest]) largest = r;
        if (largest == i) break;
        swap(a[i], a[largest]);
        i = largest;
    }
}

void heapSort(vector<int>& a) {
    int n = a.size();
    for (int i = n/2 - 1; i >= 0; --i) heapify(a, i, n);   // O(N) 建堆
    for (int i = n - 1; i > 0; --i) {
        swap(a[0], a[i]);
        heapify(a, 0, i);
    }
}
```

要点：
- **空间 O(1)**：原地，不需要额外数组（递归改成迭代时）
- **不稳定**：相等元素位置可能变
- 详见 [[heap]]

## 内省排序（introsort）— C++ `std::sort`

C++ STL `std::sort` 实际是 **introsort**：

- 默认走快排
- 递归深度过深就切换到堆排（防 O(N²) 最坏）
- 小规模（< 16）切换到插入排序（常数小）

所以 `std::sort` 严格 O(N log N)，但**不稳定**。要稳定用 `std::stable_sort`（归并版）。

## 排序题常见用法

### 1. 自定义比较

```cpp
sort(a.begin(), a.end(), [](const Item& x, const Item& y) {
    if (x.score != y.score) return x.score > y.score;     // 分高的在前
    return x.name < y.name;                                // 同分按名字字典序
});
```

**陷阱**：比较函数必须满足**严格弱序**（`<` 而非 `<=`），不然 std::sort 可能 segfault。

### 2. 部分排序（nth_element）

```cpp
nth_element(a.begin(), a.begin() + k, a.end());
// a[k] 是排序后第 k 位的值；左边都 ≤ 右边都 ≥；其余无序
```

平均 O(N)。求**第 K 小**的最快方法之一，详见 [[problems/lis|（无关，例子）]]。

### 3. 区间排序

```cpp
sort(a.begin() + l, a.begin() + r);   // 排 [l, r)
```

### 4. 快速选择（Quickselect）

求第 K 小，平均 O(N)：

```cpp
int kthSmallest(vector<int>& a, int k) {
    nth_element(a.begin(), a.begin() + k - 1, a.end());
    return a[k - 1];
}
```

## 何时用哪种

| 场景 | 推荐 |
|---|---|
| 通用 | `std::sort`（introsort） |
| 需要稳定 | `std::stable_sort`（归并） |
| 链表 | 归并（in-place 链表归并 O(1) 额外空间） |
| 求第 K | `std::nth_element`（quickselect） |
| 值域小（< N） | 计数 / 桶 |
| 整数大批量 | 基数 |
| 几乎已序 | 插入（O(N) 最优） |
| 严格 O(N log N) 上界 | 归并 / 堆 |

## 经典题型

| 题 | 用法 |
|---|---|
| 排序数组 LC 912 | 任选一种实现 |
| 颜色分类 LC 75 | **三向切分**（荷兰国旗） |
| 数组中第 K 大 LC 215 | 快速选择 / 堆 |
| 最大数 LC 179 | 自定义比较 + 排序 |
| 合并区间 LC 56 | 排序 + 扫描 |
| 数组中的逆序对 | 归并副产品 |
| 链表排序 LC 148 | 归并（O(1) 空间） |
| 摆动排序 II LC 324 | 排序 + 穿插 |

## 易错点

- 自定义比较器写成 `≤` → 严格弱序违反 → UB
- 排序 `vector<vector<int>>` 默认按字典序——是不是你想要的？
- `sort` 的 `last` 是**开**区间：`sort(a.begin(), a.end())` 包含到末位
- 大数组**栈上**声明 `int a[1e7]` 会爆栈，要 `vector` 或 `static`
- 浮点数比较容易出 NaN → 比较函数要 robust

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「排序」相关关键词（排序, 快排, 归并, 堆排, 基数）的笔记，按来源分组。

### 左程云
- [[基数排序的没有桶的优化版本-难点解释|课程/左程云/001. 难点重点照顾/01. 难点释疑/基数排序的没有桶的优化版本-难点解释.md]]
- [[工程上对排序的改进|课程/左程云/001. 难点重点照顾/工程上对排序的改进.md]]
- [[排序算法的传递性|课程/左程云/002. 知识点总结/10. 数学基础/排序算法的传递性.md]]
- [[外排序|课程/左程云/002. 知识点总结/20. 算法知识点/外排序.md]]
- [[归并排序|课程/左程云/002. 知识点总结/20. 算法知识点/归并排序.md]]
- [[拓扑排序DFS解法|课程/左程云/002. 知识点总结/20. 算法知识点/拓扑排序DFS解法.md]]

### labuladong
- [[图拓扑排序 和 并查集|课程/labuladong/经典暴力搜索算法/图拓扑排序 和 并查集.md]]

### C++算法
- [[基数排序|课程/C++算法/必备/基数排序.md]]
- [[归并|课程/C++算法/必备/归并.md]]
- [[快排|课程/C++算法/必备/快排.md]]
- [[排序总结|课程/C++算法/必备/排序总结.md]]

### 题目
- [[0033-搜索旋转排序数组|题目/0033-搜索旋转排序数组.md]]
- [[0034-在排序数组中查找元素的第一个和最后一个位置|题目/0034-在排序数组中查找元素的第一个和最后一个位置.md]]
- [[0082-删除排序链表中的重复元素 II|题目/0082-删除排序链表中的重复元素 II.md]]
- [[0083-删除排序链表中的重复元素|题目/0083-删除排序链表中的重复元素.md]]
- [[0153-寻找旋转排序数组中的最小值|题目/0153-寻找旋转排序数组中的最小值.md]]
- [[0154-寻找旋转排序数组中的最小值 II|题目/0154-寻找旋转排序数组中的最小值 II.md]]

## 参考资料

- [[../../课程/C++算法/必备/快排]] — 你的快排笔记
- [[../../课程/C++算法/必备/归并]] — 你的归并笔记
- [[../../课程/左程云/100. 算法课程]] — 左程云排序章节
- 《算法导论》第 6 / 7 / 8 章

## 关联条目

- 上层：[[algorithm-overview]]
- 思维：[[divide-and-conquer]] · [[heap]]
- 应用：[[binary-search]]（要先排序）
