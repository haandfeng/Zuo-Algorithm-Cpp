---
title: 数位 DP
aliases: [Digit DP]
tags: [pattern, dp]
created: 2026-05-03
updated: 2026-05-03
---

# 数位 DP

> **TL;DR**：求 "**[1, N] 中满足某性质的数字个数**"，按**数字的每一位**做 DP。状态通常包括"位置 + tight 标记 + 业务状态"，记忆化递归实现。

## 通用模板

```cpp
string s;            // 数字 N 转成字符串
int memo[20][...];   // memo[pos][business_state]，注意 tight 不计入 memo

int dfs(int pos, int business_state, bool tight, bool started) {
    if (pos == (int)s.size()) return is_valid(business_state) ? 1 : 0;
    if (!tight && started && memo[pos][business_state] != -1)
        return memo[pos][business_state];

    int up = tight ? s[pos] - '0' : 9;
    int ans = 0;
    for (int d = 0; d <= up; ++d) {
        int new_started = started || (d > 0);
        int new_state = transition(business_state, d, new_started);
        bool new_tight = tight && (d == up);
        ans += dfs(pos + 1, new_state, new_tight, new_started);
    }
    if (!tight && started) memo[pos][business_state] = ans;
    return ans;
}

int count(long long N) {
    s = to_string(N);
    memset(memo, -1, sizeof(memo));
    return dfs(0, initial_state, true, false);
}
```

## 关键点

### 1. tight（贴边）

`tight = true` 表示前面所有位都和 N 对应位一样；这一位的上限只能是 `s[pos]`。
**tight 状态不能被记忆化**——它强依赖具体的 N。

### 2. started（前导零）

如果题目对前导零敏感（如"含数字 7 的"中"007" 不算"含 7 的三位数"），需要 `started` 标记。

### 3. business_state

业务相关的状态——例如：
- "上一位是什么数字"（求不含连续 49 的数）
- "已经含了多少 7"
- "数字之和的 mod K"

状态数量决定记忆化数组大小，要尽量小。

## 经典题：不含 7 的数字个数

求 `[1, N]` 中**不含数字 7** 的个数。

```cpp
int memo[20];
string s;
int dfs(int pos, bool tight, bool started) {
    if (pos == (int)s.size()) return started ? 1 : 0;
    if (!tight && started && memo[pos] != -1) return memo[pos];
    int up = tight ? s[pos] - '0' : 9;
    int ans = 0;
    for (int d = 0; d <= up; ++d) {
        if (d == 7) continue;
        ans += dfs(pos + 1, tight && (d == up), started || (d > 0));
    }
    if (!tight && started) memo[pos] = ans;
    return ans;
}

int countNoSeven(long long N) {
    s = to_string(N);
    memset(memo, -1, sizeof(memo));
    return dfs(0, true, false);
}
```

## 经典题：Windy 数（HDU 1142）

定义 Windy 数 = 相邻两位数字之差 ≥ 2 的数。求 [A, B] 中 Windy 数个数。

业务状态 = "上一位是什么数字" → memo[20][10]。

## 何时想到数位 DP

| 信号 | 例子 |
|---|---|
| "[1, N] 中含 / 不含某数字 / 模式的个数" | 数位 DP 经典 |
| "数位之和 / 各位 mod K" | 数位 DP |
| "1 ~ N 二进制表示中…" | LC 600 不含连续 1 |
| N 极大（10^18） | 数位 DP（按位逐位 → log N) |

## 经典题型

| 题 | 业务状态 |
|---|---|
| 不含 7 的数 | 无 |
| Windy 数 | 上一位 |
| 不含连续 1 的二进制数 LC 600 | 上一位 |
| 数字 1 出现的次数 LC 233 | 计数（需特殊技巧） |
| 至少一位重复的数字 LC 1012 | mask 已用数字 |
| 数位计数（POJ） | mod K |

## 复杂度

**数位 DP 的好处**：把"枚举 N 个数"的 O(N) 降到"按位 DP"的 **O(log N · 状态数)**。
N = 10^18 时，O(60 · 10) = 600 → 完全可行。

## 易错点

- **tight** 状态不能记忆化
- **started** 状态对前导零敏感的题必须有
- 业务状态设计错 → 多算 / 漏算
- `[A, B]` 求范围：用 `count(B) - count(A - 1)`
- 区分"位上的数字"和"位置索引"
- 数据范围用 `long long`

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「数位DP」相关关键词（数位DP, 数位 DP）的笔记，按来源分组。

### 左程云
- [[数位DP|课程/左程云/000.Pending/数位DP.md]]
- [[数位DP|课程/左程云/002. 知识点总结/20. 算法知识点/数位DP.md]]

## 参考资料

- 《算法竞赛进阶指南》数位 DP 章节
- 灵神《数位 DP 模板讲解》

## 关联条目

- 上层：[[dp]] · [[dp-roadmap]]
- 兄弟：[[dp-knapsack]] · [[dp-interval]] · [[dp-tree]] · [[dp-bitmask]]
- 工具：[[../algorithms/bit-manipulation]]（二进制数位 DP）
