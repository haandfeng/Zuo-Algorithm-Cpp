---
title: 树状数组
aliases: [Fenwick Tree, BIT, Binary Indexed Tree]
tags: [data-structure, tree]
created: 2026-05-03
updated: 2026-05-03
---

# 树状数组（Fenwick Tree / BIT）

> **TL;DR**：用一个数组实现**单点修改 + 前缀和查询都是 O(log N)**。代码极短（10 行），是[[segment-tree|线段树]]在"求和" 场景下的轻量替代。核心是 [[../algorithms/bit-manipulation#4. lowbit：最低位 1 对应的值|lowbit]] 操作。

## 模板（最常见）

```cpp
class BIT {
    vector<long long> c;
    int n;
public:
    BIT(int n) : n(n), c(n + 1, 0) {}

    void update(int i, long long v) {           // a[i] += v
        for (; i <= n; i += i & -i) c[i] += v;
    }

    long long query(int i) {                     // sum of a[1..i]
        long long s = 0;
        for (; i > 0; i -= i & -i) s += c[i];
        return s;
    }

    long long range(int l, int r) {              // sum of a[l..r]
        return query(r) - query(l - 1);
    }
};
```

**注意**：树状数组**下标从 1 开始**，0 是哨兵（`i & -i = 0` 会死循环）。

## lowbit 是什么

`x & -x` 取出 `x` 的二进制表示中**最低的那个 1 对应的值**。

例如：

```text
x = 0b1100  (= 12)
-x = 0b...0100   (补码)
x & -x = 0b0100   (= 4)
```

意义：**c[i] 负责的区间长度** = lowbit(i)。

例如 N = 8：

```text
c[1]: 管 a[1]                (长度 1)
c[2]: 管 a[1..2]             (长度 2)
c[3]: 管 a[3]                (长度 1)
c[4]: 管 a[1..4]             (长度 4)
c[5]: 管 a[5]                (长度 1)
c[6]: 管 a[5..6]             (长度 2)
c[7]: 管 a[7]                (长度 1)
c[8]: 管 a[1..8]             (长度 8)
```

`update(i)` 沿着 `i += lowbit(i)` 往上跳——每个跳到的位置 c[j] 都管含 i 的区间。
`query(i)` 沿着 `i -= lowbit(i)` 往下跳——每次去掉一段已被加过的区间。

## 复杂度

| 操作 | 复杂度 |
|---|---|
| 建树 | O(N log N) 或 O(N) |
| 单点修改 | O(log N) |
| 前缀和查询 | O(log N) |
| 区间和查询 | O(log N)（两次前缀差） |
| 空间 | O(N) |

## 树状数组 vs 线段树

| 维度 | 树状数组 | 线段树 |
|---|---|---|
| 代码 | 10 行 | 50+ 行 |
| 常数 | 小 | 较大 |
| 可做操作 | **可拆分**（和、xor、cnt） | 任意结合操作 |
| 区间修改 | 需差分技巧 | 懒标记原生支持 |
| 区间最值 | 难做 | 直接 |
| 二维 | 容易 | 麻烦 |

**经验**：能用树状数组就用，写得快、跑得快；只有"区间最值"或"复杂区间修改"才必须线段树。

## 区间修改 + 单点查询（差分版）

用 `bit` 维护差分数组的前缀和：

```cpp
// 区间 [l, r] 加 v
bit.update(l, v);
bit.update(r + 1, -v);

// 单点查询 a[i]
long long v = bit.query(i);
```

## 区间修改 + 区间查询

需要两个 BIT，公式稍复杂：

```cpp
// 设 d[i] = a[i] - a[i-1] 是差分
// sum(1..i) = sum_{j=1..i} (i - j + 1) * d[j]
//           = (i + 1) * sum(d[1..i]) - sum(j * d[j], j=1..i)

class BIT2 {
    BIT b1, b2;
public:
    BIT2(int n) : b1(n), b2(n) {}
    void rangeUpdate(int l, int r, long long v) {
        b1.update(l, v);
        b1.update(r + 1, -v);
        b2.update(l, v * l);
        b2.update(r + 1, -v * (r + 1));
    }
    long long prefixSum(int i) {
        return (i + 1) * b1.query(i) - b2.query(i);
    }
    long long rangeSum(int l, int r) {
        return prefixSum(r) - prefixSum(l - 1);
    }
};
```

## 经典题型

| 题 | 用法 |
|---|---|
| 区域和检索 - 数组可修改 LC 307 | 模板 |
| 计算右侧小于当前元素的个数 LC 315 | BIT + 离散化 |
| 翻转对 LC 493 | 归并 / BIT |
| 区间和的个数 LC 327 | BIT + 离散化 |
| 求逆序对数 | 离散化 + BIT |

### 例：求逆序对

```cpp
long long countInversions(vector<int>& a) {
    // 离散化
    auto sorted = a;
    sort(sorted.begin(), sorted.end());
    sorted.erase(unique(sorted.begin(), sorted.end()), sorted.end());
    auto rk = [&](int x) {
        return lower_bound(sorted.begin(), sorted.end(), x) - sorted.begin() + 1;
    };
    int m = sorted.size();
    BIT bit(m);
    long long ans = 0;
    for (int i = (int)a.size() - 1; i >= 0; --i) {
        int r = rk(a[i]);
        ans += bit.query(r - 1);     // 已加入的 < a[i] 的个数
        bit.update(r, 1);
    }
    return ans;
}
```

## 易错点

- **下标从 1 开始**——0 会无限循环
- 数组大小开够（用 `n + 1` 而不是 `n`）
- 区间修改 + 区间查询 → 用双 BIT 公式，别和单 BIT 混
- 离散化时 `lower_bound` 别忘 `+ 1`（要 1-indexed）
- `long long` 防溢出

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「树状数组」相关关键词（树状数组, Fenwick, BIT）的笔记，按来源分组。

### 左程云
- [[树状数组|课程/左程云/002. 知识点总结/20. 算法知识点/树状数组.md]]

## 参考资料

- [[../../课程/左程云/100. 算法课程]] — 左程云树状数组章节
- 《算法竞赛进阶指南》BIT 章节
- Peter Fenwick, *A New Data Structure for Cumulative Frequency Tables*, 1994

## 关联条目

- 上层：[[ds-overview]]
- 兄弟：[[segment-tree]] · [[bst]]
- 工具：[[../algorithms/bit-manipulation]]（lowbit 的来源）
