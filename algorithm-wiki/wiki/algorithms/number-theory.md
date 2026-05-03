---
title: 数论基础
aliases: [Number Theory, GCD, 快速幂, 素数筛, 逆元]
tags: [algorithm, math]
created: 2026-05-03
updated: 2026-05-03
---

# 数论基础

> **TL;DR**：算法竞赛 / 面试常用的数论"四件套"——**GCD**、**快速幂**、**素数筛**、**逆元**。

## 1. GCD（最大公约数）

```cpp
int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}
// C++17 可直接用 std::gcd / std::lcm
int g = std::gcd(a, b);
int l = std::lcm(a, b);     // = a / gcd(a,b) * b
```

时间 O(log min(a, b))。**裴蜀定理**：`ax + by = c` 有整数解 ⟺ `gcd(a, b) | c`。

## 2. 快速幂

```cpp
long long quickPow(long long a, long long n, long long mod) {
    long long res = 1;
    a %= mod;
    while (n > 0) {
        if (n & 1) res = res * a % mod;
        a = a * a % mod;
        n >>= 1;
    }
    return res;
}
```

时间 O(log N)。**几乎所有涉及幂取模 / 大数乘法**的题都用得上。

矩阵快速幂：把数换成矩阵，斐波那契、线性递推都能 O(log N) 解。

## 3. 素数筛

埃拉托色尼筛 O(N log log N)：

```cpp
vector<int> sieve(int n) {
    vector<bool> isPrime(n + 1, true);
    vector<int> primes;
    isPrime[0] = isPrime[1] = false;
    for (int i = 2; i <= n; ++i) {
        if (isPrime[i]) {
            primes.push_back(i);
            for (long long j = (long long)i * i; j <= n; j += i)
                isPrime[j] = false;
        }
    }
    return primes;
}
```

线性筛 O(N)：每个合数被它**最小质因数**筛一次。

## 4. 逆元（模质数）

求 `a^(-1) mod p`：

```cpp
// 费马小定理：p 是质数且 gcd(a, p) = 1
long long inv(long long a, long long p) {
    return quickPow(a, p - 2, p);
}
```

或扩展欧几里得。常用于**组合数取模**。

## 经典题型

| 题 | 用法 |
|---|---|
| 快速幂 | 模板 |
| 计算第 n 个斐波那契数 mod 1e9+7 | 矩阵快速幂 |
| 统计区间内的素数 | 素数筛 |
| 组合数 C(n, k) mod p | 阶乘 + 逆元 |
| 中国剩余定理 | 解同余方程组 |
| 欧拉函数 | 数论函数 |

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「数论」相关关键词（数论, GCD, 素数, 快速幂）的笔记，按来源分组。

### 左程云
- [[素数筛法|课程/左程云/002. 知识点总结/10. 数学基础/素数筛法.md]]
- [[快速幂|课程/左程云/004. 刷题经验总结/快速幂.md]]
- [[26 由斐波那契数列讲述矩阵快速幂技巧|课程/左程云/100. 算法课程/01. 体系学习班/26 由斐波那契数列讲述矩阵快速幂技巧.md]]

## 关联条目

- 上层：[[../maps/algorithm-overview]]
- 工具：[[bit-manipulation]]
