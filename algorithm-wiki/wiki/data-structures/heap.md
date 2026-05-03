---
title: 堆 / 优先队列
aliases: [Heap, Priority Queue, 优先队列, 二叉堆]
tags: [data-structure, tree]
created: 2026-05-03
updated: 2026-05-03
---

# 堆 / 优先队列

> **TL;DR**：堆是一种**特殊完全二叉树**：每个节点都 ≥（大顶堆）或 ≤（小顶堆）孩子。push / pop O(log N)，看堆顶 O(1)。**Top-K** 和 **Dijkstra** 的核心数据结构。

## 数组表示

完全二叉树天然适合用数组存（下标从 0 开始）：

```text
索引：     0  1  2  3  4  5  6
值：       9  7  6  5  3  4  2

           9
          / \
         7   6
        / \ / \
       5  3 4  2

父节点 i 的左孩子 = 2i + 1，右孩子 = 2i + 2
孩子 i 的父节点 = (i - 1) / 2
```

## 两个核心操作

### 1. up（上浮）— push 时用

新元素放数组末尾，然后**与父比较**，比父大（大顶堆）就上浮：

```cpp
void up(vector<int>& heap, int i) {
    while (i > 0) {
        int parent = (i - 1) / 2;
        if (heap[i] <= heap[parent]) break;
        swap(heap[i], heap[parent]);
        i = parent;
    }
}
```

### 2. down（下沉）— pop 时用

把根换成末尾元素，再**与较大的孩子比较**，比孩子小就下沉：

```cpp
void down(vector<int>& heap, int i, int n) {
    while (true) {
        int l = 2 * i + 1, r = 2 * i + 2, largest = i;
        if (l < n && heap[l] > heap[largest]) largest = l;
        if (r < n && heap[r] > heap[largest]) largest = r;
        if (largest == i) break;
        swap(heap[i], heap[largest]);
        i = largest;
    }
}
```

每次 up / down 最多走 O(log N) 层。

## C++ 的优先队列

```cpp
// 大顶堆（默认）
priority_queue<int> pq;

// 小顶堆
priority_queue<int, vector<int>, greater<int>> pq;

// 自定义比较：按 pair 第二个升序（小顶堆）
priority_queue<pair<int,int>,
               vector<pair<int,int>>,
               greater<pair<int,int>>> pq;

// 自定义结构体
struct Cmp {
    bool operator()(const Node& a, const Node& b) const {
        return a.dist > b.dist;       // 注意：> 是小顶堆
    }
};
priority_queue<Node, vector<Node>, Cmp> pq;

// 操作
pq.push(x);            // 入
pq.top();              // 看堆顶
pq.pop();              // 弹堆顶
pq.size();
pq.empty();
```

## 复杂度

| 操作 | 复杂度 |
|---|---|
| 看堆顶 | O(1) |
| push | O(log N) |
| pop | O(log N) |
| **建堆**（Floyd 法） | **O(N)** |
| 删除任意元素 | O(N)（要先找到） |
| 修改任意元素的值 | O(log N)（已持下标）+ 找下标的代价 |

**建堆 O(N)** 是个反直觉但重要的事实。从最后一个非叶子节点向上做 down：

```cpp
for (int i = n / 2 - 1; i >= 0; --i) down(heap, i, n);
```

数学上每层节点数指数减少、每个节点 down 的步数是它到叶子的距离 → 总和是等比级数 → O(N)。

## 何时想到堆

| 信号 | 用法 |
|---|---|
| "前 K 大 / 小" | 维护一个大小为 K 的反向堆 |
| "数据流中位数" | 大顶堆 + 小顶堆双堆 |
| "合并 K 个有序" | 把每个序列头放进小顶堆 |
| "调度 / 优先级任务" | 优先队列 |
| "Dijkstra 最短路" | 小顶堆 |
| "霍夫曼编码" | 每次取两个最小合并 |
| "贪心 + 排序后还要动态最小" | 小顶堆 |

## 经典题

| 题 | 用法 |
|---|---|
| Top-K 最大 LC 215 | 维护大小 K 的**小顶堆** |
| 合并 K 个有序链表 LC 23 | 小顶堆装节点 |
| 数据流中位数 LC 295 | 大顶堆（左半）+ 小顶堆（右半） |
| 任务调度器 LC 621 | 大顶堆 + 冷却 |
| 滑窗最大值 LC 239 | 也可以用堆，但 [[monotonic-queue\|单调队列]] 更优 |
| 第 K 小 BST LC 230 | 中序，**不是堆** |
| K 对最小和 LC 373 | 小顶堆 + lazy 扩展 |

### Top-K 最大模板

```cpp
int findKthLargest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> pq;   // 小顶堆
    for (int x : nums) {
        pq.push(x);
        if ((int)pq.size() > k) pq.pop();
    }
    return pq.top();
}
```

时间 O(N log K)，空间 O(K)。当 N 巨大、K 小的时候比排序 O(N log N) 优。

### 双堆中位数模板

```cpp
class MedianFinder {
    priority_queue<int> lo;                                // 大顶堆，存较小一半
    priority_queue<int, vector<int>, greater<int>> hi;     // 小顶堆，存较大一半
public:
    void addNum(int num) {
        lo.push(num);
        hi.push(lo.top()); lo.pop();
        if (hi.size() > lo.size()) { lo.push(hi.top()); hi.pop(); }
    }
    double findMedian() {
        return lo.size() > hi.size() ? lo.top() : (lo.top() + hi.top()) / 2.0;
    }
};
```

## 堆排序

把整个数组建堆 O(N)，然后**反复 pop**得排好序的序列：

```cpp
void heapSort(vector<int>& a) {
    int n = a.size();
    for (int i = n / 2 - 1; i >= 0; --i) down(a, i, n);    // 建大顶堆
    for (int i = n - 1; i > 0; --i) {
        swap(a[0], a[i]);                                   // 把最大放到尾部
        down(a, 0, i);                                      // 调整剩余 [0, i)
    }
}
```

时间 O(N log N) 严格上界，空间 O(1)。详见 [[sorting]]。

## 易错点

- 大顶堆 vs 小顶堆**比较函数方向**容易反：`less<T>` 是大顶堆，`greater<T>` 是小顶堆
- C++ `pop()` 不返回值
- 自定义结构体比较函数符号搞反 → 全错（特别是 lambda）
- 堆**不能**做"找前驱后继 / 中间元素" → 那是 [[bst]] 的活
- 堆里**修改任意值**很贵 → 通常用"懒删除"标记，pop 时跳过失效项

## 参考资料

- [[../../课程/C++算法/必备/堆 优先队列]] — 你的堆专题笔记
- [[../../语言/C++/STL]] — `std::priority_queue`

## 关联条目

- 上层：[[ds-overview]] · [[binary-tree]]
- 兄弟：[[bst]] · [[queue]]
- 应用：[[sorting]] · [[shortest-path]]
