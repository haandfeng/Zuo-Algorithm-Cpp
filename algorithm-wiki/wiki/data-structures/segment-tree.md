---
title: 线段树
aliases: [Segment Tree, 区间树]
tags: [data-structure, tree]
created: 2026-05-03
updated: 2026-05-03
---

# 线段树

> **TL;DR**：线段树用一棵**完全二叉树**维护数组的**区间信息**（和 / 最值 / GCD…），支持 **O(log N) 单点修改 + O(log N) 区间查询**。配合**懒标记**还能做 **O(log N) 区间修改**。

## 何时用

朴素数组：

- 区间查询 O(N)
- 区间修改 O(N)

[[fenwick-tree|树状数组]]：

- 区间和 O(log N)、单点修改 O(log N)
- **不擅长**最值、其它非可拆分操作

线段树：

- 任意可结合操作（max / min / sum / gcd / xor / …）
- 单点 + 区间修改 / 查询都 O(log N)
- 实现稍长，常数较大

## 数组表示（完全二叉树）

数组下标从 1 开始：

- 根节点 = 1
- 节点 i 的左孩子 = 2i、右孩子 = 2i + 1
- 总空间 4N 足够

```cpp
struct SegTree {
    int n;
    vector<long long> tree, lazy;

    SegTree(int n) : n(n), tree(4 * n, 0), lazy(4 * n, 0) {}

    void build(vector<int>& a, int node, int l, int r) {
        if (l == r) { tree[node] = a[l]; return; }
        int mid = (l + r) / 2;
        build(a, 2*node, l, mid);
        build(a, 2*node+1, mid+1, r);
        tree[node] = tree[2*node] + tree[2*node+1];
    }
};
```

## 核心三操作（区间求和 + 区间加）

### 单点修改 / 区间查询（无懒标记）

```cpp
// 单点 +v
void update(int node, int l, int r, int idx, int v) {
    if (l == r) { tree[node] += v; return; }
    int mid = (l + r) / 2;
    if (idx <= mid) update(2*node, l, mid, idx, v);
    else            update(2*node+1, mid+1, r, idx, v);
    tree[node] = tree[2*node] + tree[2*node+1];
}

// 查询 [ql, qr] 之和
long long query(int node, int l, int r, int ql, int qr) {
    if (qr < l || r < ql) return 0;
    if (ql <= l && r <= qr) return tree[node];
    int mid = (l + r) / 2;
    return query(2*node, l, mid, ql, qr) + query(2*node+1, mid+1, r, ql, qr);
}
```

### 懒标记：区间修改

```cpp
void push_down(int node, int l, int r) {
    if (lazy[node] == 0) return;
    int mid = (l + r) / 2;
    tree[2*node]   += lazy[node] * (mid - l + 1);
    tree[2*node+1] += lazy[node] * (r - mid);
    lazy[2*node]   += lazy[node];
    lazy[2*node+1] += lazy[node];
    lazy[node] = 0;
}

void rangeUpdate(int node, int l, int r, int ql, int qr, long long v) {
    if (qr < l || r < ql) return;
    if (ql <= l && r <= qr) {
        tree[node] += v * (r - l + 1);
        lazy[node] += v;
        return;
    }
    push_down(node, l, r);
    int mid = (l + r) / 2;
    rangeUpdate(2*node, l, mid, ql, qr, v);
    rangeUpdate(2*node+1, mid+1, r, ql, qr, v);
    tree[node] = tree[2*node] + tree[2*node+1];
}

long long rangeQuery(int node, int l, int r, int ql, int qr) {
    if (qr < l || r < ql) return 0;
    if (ql <= l && r <= qr) return tree[node];
    push_down(node, l, r);
    int mid = (l + r) / 2;
    return rangeQuery(2*node, l, mid, ql, qr) +
           rangeQuery(2*node+1, mid+1, r, ql, qr);
}
```

## 复杂度

| 操作 | 复杂度 |
|---|---|
| 建树 | O(N) |
| 单点修改 | O(log N) |
| 区间修改（懒标记） | O(log N) |
| 区间查询 | O(log N) |
| 空间 | O(4N) |

## 何时想到线段树

| 信号 | 用法 |
|---|---|
| "动态区间最值 / 求和" | 线段树 |
| "区间加 + 区间求和" | 线段树 + 懒标记 |
| "区间染色" | 线段树（节点存颜色） |
| "动态前驱 / 后继" | 线段树二分 |
| "区间最值 + 单点修改" | 线段树（树状数组难做） |

## 经典题型

| 题 | 用法 |
|---|---|
| 区域和检索 - 数组可修改 LC 307 | 树状数组 / 线段树 |
| 计算右侧小于当前元素的个数 LC 315 | 线段树 + 离散化 |
| 区间和的个数 LC 327 | 线段树 / 归并 |
| 我的日程安排表 III LC 732 | 线段树 / map 差分 |
| 矩形面积 II LC 850 | 扫描线 + 线段树 |
| 落下的方块 LC 699 | 线段树（区间最值） |

## 线段树的变种

- **可持久化线段树**（主席树）：维护历史版本
- **二维线段树**：矩阵区间操作
- **动态开点线段树**：值域大但稀疏
- **吉如一线段树**：复杂区间最值更新

## 易错点

- 数组开 4N 而不是 N
- 下标从 1 开始还是 0 开始：与公式一致
- 区间表示 `[l, r]` 闭区间，**别用半开**
- 懒标记下传顺序：**先下传再递归**
- 累加型懒标记 vs 覆盖型懒标记，混了就错
- 长整型溢出 → 用 `long long`

## 参考资料

- [[../../课程/左程云/100. 算法课程]] — 左程云线段树章节
- 《算法竞赛进阶指南》线段树章节

## 关联条目

- 上层：[[ds-overview]]
- 兄弟：[[fenwick-tree]] · [[bst]]
- 衍生：[[../techniques/scan-line]]（扫描线 + 线段树）
