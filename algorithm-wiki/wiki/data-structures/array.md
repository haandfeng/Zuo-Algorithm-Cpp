---
title: 数组
aliases: [Array, 顺序表, 动态数组, vector]
tags: [data-structure, linear]
created: 2026-05-03
updated: 2026-05-03
---

# 数组

> **TL;DR**：数组是**连续内存 + 整数下标**的最基础线性结构。访问 O(1)，中间插删 O(N)。是几乎所有其它数据结构的底层实现砖块。

## 静态 vs 动态

```cpp
int a[100];          // 静态数组：编译期定大小、栈上
vector<int> b;        // 动态数组：运行期可扩、堆上
int* c = new int[n];  // 手动堆上数组（C 风格，少用）
```

**动态数组扩容**机制：

- 容量满了就**翻倍**（C++ vector 实现一般 1.5 或 2 倍）
- 扩容时 memcpy 整块，单次 O(N)
- 但 N 次 push_back 总 O(N) → **均摊 O(1)**（见 [[complexity#均摊复杂度]]）

## 复杂度速查

| 操作 | 复杂度 |
|---|---|
| `a[i]` 读写 | O(1) |
| 末尾 push_back | 均摊 O(1) |
| 末尾 pop_back | O(1) |
| 中间插入 / 删除 | O(N)（要挪后面） |
| 查找特定值 | O(N)（无序）/ O(log N)（有序 + 二分） |
| 排序 | O(N log N) |

## 二维数组

```cpp
vector<vector<int>> mat(n, vector<int>(m, 0));   // n × m 全零
int mat[100][100];                                // 静态二维
```

**内存布局**是**行优先**（C/C++）：`mat[i][j]` 物理上 = `mat` 起始 + `i * m + j` 个 int。
所以**先遍历 j 再遍历 i 比反过来快**（cache-friendly）：

```cpp
for (int i = 0; i < n; ++i)            // 慢的写法是 i,j 互换
    for (int j = 0; j < m; ++j)
        mat[i][j] = ...;
```

## 滚动数组（DP 优化）

二维 DP 经常只依赖**上一行**，可以滚动：

```cpp
// 完整二维 dp[i][j]
vector<vector<int>> dp(n + 1, vector<int>(m + 1));
for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= m; ++j)
        dp[i][j] = dp[i-1][j] + dp[i][j-1];

// 滚动到一维
vector<int> dp(m + 1);
for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= m; ++j)
        dp[j] = dp[j] + dp[j-1];   // dp[j] 是上一行，dp[j-1] 是当前行
```

空间从 O(NM) 降到 O(M)。

## 常见用法

### 1. 当作哈希表（值域小）

如果元素取值范围小（如 0~25），用数组比哈希表快得多：

```cpp
int cnt[26] = {0};
for (char c : s) cnt[c - 'a']++;
```

### 2. 前缀和 / 差分

详见 [[prefix-sum]]。

### 3. 与双指针组合

详见 [[two-pointers]]、[[sliding-window]]。

### 4. 原地操作

很多题要求 O(1) 额外空间，例如：

- LC 27 移除元素：双指针原地覆盖
- LC 73 矩阵置零：用第一行第一列当标记数组
- LC 41 缺失的第一个正数：把 a[i] 放到下标 a[i]-1 上

## 易错点

- **下标越界**：`a[n]` 在长度为 n 的数组里越界。永远 `i < n` 不要 `i <= n`
- **二维数组声明**：`int a[3][4]` 和 `int a[4][3]` 不一样
- **vector 默认初始值**：`vector<int>(n)` 全 0；`vector<int>(n, -1)` 才是 -1
- **传引用还是传值**：大数组传值会拷贝，要 `void f(vector<int>& a)`

## 参考资料

- [[../../课程/C++算法/入门笔记]] — 入门笔记 1~3 节
- [[../../语言/C++/STL]] — vector 的使用

## 关联条目

- 上层：[[ds-overview]]
- 兄弟：[[linked-list]]
- 技巧：[[two-pointers]] · [[sliding-window]] · [[prefix-sum]]
- 应用：[[binary-search]] · [[sorting]]
