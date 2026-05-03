---
title: 位运算
aliases: [Bit Manipulation, 二进制, bitmask, 位操作]
tags: [algorithm, math]
created: 2026-05-03
updated: 2026-05-03
---

# 位运算

> **TL;DR**：位运算 = 直接操作整数的二进制位。常用于**集合表示、快速乘除、状态压缩、判奇偶、奇技淫巧**。是 C++ / 算法竞赛 / 嵌入式必备技能。

## 6 个基本操作

| 操作 | 符号 | 例 |
|---|---|---|
| 与 | `&` | `0b1100 & 0b1010 = 0b1000` |
| 或 | `|` | `0b1100 | 0b1010 = 0b1110` |
| 异或 | `^` | `0b1100 ^ 0b1010 = 0b0110` |
| 非 | `~` | `~0b1100 = ...11110011`（按 int 长度） |
| 左移 | `<<` | `0b0011 << 2 = 0b1100` |
| 右移 | `>>` | `0b1100 >> 2 = 0b0011` |

## 常用技巧（必背）

### 1. 取/设/翻/清第 k 位

```cpp
bit_k = (x >> k) & 1;          // 取第 k 位
x |=  (1 << k);                // 设第 k 位为 1
x ^=  (1 << k);                // 翻转第 k 位
x &= ~(1 << k);                // 清第 k 位为 0
```

### 2. 判奇偶

```cpp
if (x & 1) odd; else even;     // 比 x % 2 快
```

### 3. 判 2 的幂

```cpp
bool isPow2 = x > 0 && (x & (x - 1)) == 0;
```

`x - 1` 把最低位的 1 变 0、之后的 0 变 1。如果 x 只有一个 1，与之后就是 0。

### 4. lowbit：最低位 1 对应的值

```cpp
int lb = x & (-x);              // 树状数组的核心操作
```

例：`x = 0b10100`，`-x = 0b01100`（补码），`x & -x = 0b00100`。

### 5. 数 1 的个数（popcount）

```cpp
int cnt = __builtin_popcount(x);     // GCC 内建，O(1)
int cnt = __builtin_popcountll(x);    // long long 版

// 手写：Brian Kernighan
int popcount(int x) {
    int cnt = 0;
    while (x) { x &= x - 1; cnt++; }     // 每次去掉最低位 1
    return cnt;
}
```

### 6. 异或的几个性质

- `a ^ a = 0`
- `a ^ 0 = a`
- 异或满足交换律 + 结合律

应用：**找数组中只出现一次的数**（其它都出现两次）：所有数异或起来即答案，O(N) 时间 O(1) 空间。LC 136。

### 7. 交换两数（无临时变量）

```cpp
a ^= b;
b ^= a;
a ^= b;
```

(实战不推荐——可读性差，编译器对 `swap` 已经优化得很好。)

### 8. 不用加号的加法

```cpp
int add(int a, int b) {
    while (b) {
        int carry = (a & b) << 1;
        a = a ^ b;
        b = carry;
    }
    return a;
}
```

LC 371。

## 状态压缩 / bitmask

用一个 int 的二进制位表示一个**集合**：

```text
集合 {0, 2, 3} 对应 0b1101 = 13
集合 {1, 4}    对应 0b10010 = 18
```

操作：

```cpp
mask | (1 << i)         // 加入元素 i
mask & ~(1 << i)        // 删除元素 i
mask & (1 << i)         // 检测元素 i 在不在
mask ^ (1 << i)         // 翻转元素 i
__builtin_popcount(mask)// 集合大小
```

枚举所有子集：

```cpp
for (int sub = mask; sub > 0; sub = (sub - 1) & mask) {
    // sub 是 mask 的非空子集
}
```

详见 [[dp-bitmask|状压 DP]]。

## 经典题型

| 题 | 用法 |
|---|---|
| 只出现一次的数字 LC 136 | 异或 |
| 只出现一次的数字 II LC 137 | 位计数 |
| 只出现一次的数字 III LC 260 | 异或 + lowbit 分组 |
| 位 1 的个数 LC 191 | popcount |
| 颠倒二进制位 LC 190 | 逐位翻 / 分治 |
| 比特位计数 LC 338 | DP: `dp[i] = dp[i >> 1] + (i & 1)` |
| 数字范围按位与 LC 201 | 公共前缀 |
| 子集 LC 78 | 枚举 `2^n` 种 mask |
| 单词长度的最大乘积 LC 318 | 字母集 → 32 位 mask |
| 两个整数之和 LC 371 | 不用加号 |
| 4 的幂 LC 342 | `(x & -x) == x && (x & 0x55555555) != 0` |
| 数组中两个数的最大异或 LC 421 | 0-1 [[trie\|Trie]] |

## 复杂度技巧

- **`__builtin_popcount`**: GCC 内建，硬件指令 O(1)
- **`__builtin_clz` / `__builtin_ctz`**: 前导 / 末尾 0 个数，O(1)
- **`__builtin_ffs`**: find first set，O(1)
- 这些函数在 `int = 0` 时是 UB，注意保护

## 与其它概念的关系

- [[fenwick-tree|树状数组]]：核心是 `lowbit`
- [[dp-bitmask|状压 DP]]：枚举子集 + 转移
- [[trie|0-1 Trie]]：处理最大异或对
- 哈希：偶尔用位运算合成 hash

## 易错点

- 优先级！`x & 1 == 1` 实际是 `x & (1 == 1) = x & 1`，**结果对但理由错**。`(x >> k) & 1 == 1` 错——要写 `((x >> k) & 1) == 1`
- 负数右移：C++ 实现定义（一般算术右移），**有符号数右移 = -1 不变**，写 bitmask 用 `unsigned`
- `1 << 31` 是 UB（int 是 32 位时）：要写 `1LL << 31` 或 `1u << 31`
- `int` 和 `long long` 混用时 mask 大小要统一

## 参考资料

- [[../../课程/C++算法/必备/位运算]] — 你的位运算笔记
- [[../../课程/左程云/002. 知识点总结]] — 左程云位运算
- *Hacker's Delight*（Henry Warren） — 经典位运算技巧大全

## 关联条目

- 上层：[[algorithm-overview]]
- 应用：[[dp-bitmask]] · [[fenwick-tree]] · [[trie]]
