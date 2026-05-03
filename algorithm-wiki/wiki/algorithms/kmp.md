---
title: KMP
aliases: [KMP, 字符串匹配, next 数组]
tags: [algorithm, string]
created: 2026-05-03
updated: 2026-05-03
---

# KMP

> **TL;DR**：KMP 是字符串匹配算法。**通过预处理模式串的"最长公共前后缀"数组（next/fail）**，匹配失败时跳过已知必然不匹配的位置，把朴素的 O(NM) 降到 **O(N + M)**。

## 问题

给文本 `text`（长 N）和模式 `pattern`（长 M），找 `pattern` 在 `text` 中**所有出现位置**。

朴素做法：每个起点都 O(M) 比较，总 O(NM)。

KMP 的核心想法：**当匹配到一半失败时，不要把模式串平移 1 格，而是利用已匹配的信息直接跳到合理位置**。

## next 数组（fail 数组 / 失败函数）

`next[i]` 定义为：**模式串前 i 个字符的"最长公共真前后缀长度"**。

例如 `pattern = "ababc"`：

| i | pattern[0..i-1] | 最长公共真前后缀 | next[i] |
|---|---|---|---|
| 0 | "" | "" | 0 |
| 1 | "a" | "" | 0 |
| 2 | "ab" | "" | 0 |
| 3 | "aba" | "a" | 1 |
| 4 | "abab" | "ab" | 2 |
| 5 | "ababc" | "" | 0 |

注意：**真前后缀**意思是不能等于整个串本身。

## 求 next 数组

经典的"自我匹配"思路：

```cpp
vector<int> getNext(const string& p) {
    int m = p.size();
    vector<int> nxt(m + 1, 0);
    int j = 0;                          // j 是当前最长公共前后缀长度
    for (int i = 2; i <= m; ++i) {
        while (j > 0 && p[j] != p[i - 1]) j = nxt[j];     // 失败回退
        if (p[j] == p[i - 1]) ++j;
        nxt[i] = j;
    }
    return nxt;
}
```

时间 O(M) —— 每次进入 while 循环 j 必减少，整体均摊。

## 主匹配过程

```cpp
vector<int> kmpSearch(const string& text, const string& pattern) {
    int n = text.size(), m = pattern.size();
    if (m == 0) return {0};
    auto nxt = getNext(pattern);
    vector<int> matches;
    int j = 0;
    for (int i = 0; i < n; ++i) {
        while (j > 0 && text[i] != pattern[j]) j = nxt[j];
        if (text[i] == pattern[j]) ++j;
        if (j == m) {
            matches.push_back(i - m + 1);
            j = nxt[j];                  // 继续找下一个
        }
    }
    return matches;
}
```

时间 O(N + M)，空间 O(M)。

## 直觉：next 跳的是什么

匹配 `text[i] != pattern[j]` 失败时：

- 朴素做法：i 回到 i - j + 1，j 回到 0
- KMP：i **不回退**，j 跳到 `next[j]`

为什么 j 跳到 next[j] 不会错过答案？因为：
- 当前已匹配 `text[i-j..i-1] == pattern[0..j-1]`
- next[j] 的定义保证 `pattern[0..next[j]-1] == pattern[j-next[j]..j-1] == text[i-next[j]..i-1]`
- 所以从 `pattern[next[j]]` 开始继续比较是合理的"最远跳跃"

## 经典应用

| 题 | 用法 |
|---|---|
| 实现 strStr() LC 28 | 直接 KMP |
| 重复的子字符串 LC 459 | next 数组判周期 |
| 最短回文串 LC 214 | 反串拼接 + KMP |
| 旋转字符串 LC 796 | 拼接两份再 KMP |
| 周期串 | next[n] != 0 且 n % (n - next[n]) == 0 |

### 例：判断字符串能否由子串重复构成

字符串 s 可以由它的某个子串重复 k 次（k ≥ 2）构成 ⟺ `n % (n - next[n]) == 0` 且 `n != next[n]`。

```cpp
bool isRepeated(string s) {
    int n = s.size();
    auto nxt = getNext(s);
    int len = n - nxt[n];          // 最小循环节长度
    return nxt[n] != 0 && n % len == 0;
}
```

## KMP vs 其它字符串匹配

| 算法 | 复杂度 | 优点 |
|---|---|---|
| 朴素 | O(NM) | 实现最简单 |
| **KMP** | O(N + M) | 经典、好理解（学过之后） |
| Z 函数 | O(N + M) | 模板更短 |
| Rabin-Karp（hash） | 平均 O(N + M) | 多模式快、好写 |
| Sunday | 平均亚线性 | 实现简单 |
| Boyer-Moore | 平均亚线性 | 文本匹配工业标准（grep） |
| AC 自动机 | O(N + M + 答案) | 多模式 |
| 后缀自动机 / 后缀数组 | O(N + M) | 复杂查询 |

实战中：

- **面试**：会 KMP 即可
- **算法题**：KMP 或字符串哈希都行
- **工程**：直接调库（C++ `string::find`、Python `str.find`），它们用的是 BM 类算法

## 易错点

- next 数组**下标含义**不一致：有的版本是"前 i 个字符"，有的是"前 i+1 个字符"。**自己定一个用法不要混**
- 求 next 时 `i` 从 2 还是 1 开始？取决于 next[0] / next[1] 是否提前设定
- 主匹配 `j == m` 时**不要 break**，要继续找下一个匹配
- 字符比较 `==` 不要写成 `=`（C++ 编译能过但是 BUG）

## 参考资料

- [[../../课程/灵茶山艾府 + 代码随想录/字符串]] — 你的字符串专题
- [[../../课程/左程云/100. 算法课程]] — 左程云 KMP 章节
- *Knuth, Morris, Pratt, Fast Pattern Matching in Strings*, 1977

## 关联条目

- 上层：[[algorithm-overview]]
- 数据结构：[[trie]]
- 兄弟：[[manacher]]
