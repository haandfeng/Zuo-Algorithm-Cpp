---
marp: true
theme: default
size: 16:9
paginate: true
header: '算法 Wiki · 二分查找模板'
footer: '基于 wiki/algorithms/binary-search.md 与 wiki/techniques/binary-search-answer.md'
---

# 二分查找：3 个模板 + 二分答案

来源：[[../../wiki/algorithms/binary-search]]、[[../../wiki/techniques/binary-search-answer]]
日期：2026-05-03

---

## 为什么需要 3 个模板

二分的 4 个易错点：

1. `(l + r) / 2` 整数溢出
2. 区间开闭混用
3. 模板背错（lower / upper / 找数）
4. 死循环

→ 准备 3 个**模式化模板**，按题型直接套。

---

## 模板 1：找等于 target

```cpp
int l = 0, r = a.size() - 1;
while (l <= r) {
    int m = l + (r - l) / 2;
    if (a[m] == target) return m;
    else if (a[m] < target) l = m + 1;
    else                    r = m - 1;
}
return -1;
```

区间 `[l, r]` 闭区间，找不到返回 -1。

---

## 模板 2：lower_bound（第一个 ≥ target）

```cpp
int l = 0, r = a.size();           // r = size，半开区间！
while (l < r) {
    int m = l + (r - l) / 2;
    if (a[m] >= target) r = m;     // m 可能就是答案
    else                l = m + 1;
}
return l;                           // 可能 == size
```

返回 size 表示找不到 ≥ target 的元素。

---

## 模板 3：upper_bound（第一个 > target）

```cpp
int l = 0, r = a.size();
while (l < r) {
    int m = l + (r - l) / 2;
    if (a[m] > target) r = m;
    else               l = m + 1;
}
return l;
```

最后一个 ≤ target 的下标 = `upper_bound - 1`。
出现次数 = `upper_bound - lower_bound`。

---

## 二分答案：核心思想

不是"在数组里二分下标"，而是"在**答案空间**里二分"。

适用条件：

- 求**最小的最大** / **最大的最小**
- 答案空间 `[lo, hi]` 有界
- 给定 mid，能 O(N) 判定"是否可行"
- 单调性：mid 可行 → mid+1 也可行（或反之）

---

## 二分答案：模板（求最小）

```cpp
int lo = …, hi = …;
while (lo < hi) {
    int mid = lo + (hi - lo) / 2;       // 求最小：下取整
    if (check(mid)) hi = mid;
    else            lo = mid + 1;
}
return lo;
```

求最大改用上取整 `(lo + hi + 1) / 2` + `lo = mid` / `hi = mid - 1`。

---

## 配题 1：Koko 吃香蕉（LC 875）

题意：n 堆香蕉、每小时最多吃 K 个、需在 H 小时内吃完。求最小 K。

```cpp
auto check = [&](int K) {
    long long h = 0;
    for (int p : piles) h += (p + K - 1) / K;
    return h <= H;
};
int lo = 1, hi = *max_element(piles.begin(), piles.end());
// ... 套模板
```

---

## 配题 2：分割数组的最大值（LC 410）

题意：把数组分 K 段，每段和的**最大值最小**。

```cpp
auto check = [&](int s) {
    int cnt = 1, cur = 0;
    for (int x : nums) {
        if (cur + x > s) { cnt++; cur = x; }
        else             cur += x;
    }
    return cnt <= K;
};
int lo = *max_element(...), hi = accumulate(...);
```

---

## 易错避坑

| 错误 | 怎么修 |
|---|---|
| 死循环 | 求最大用 `mid = (l + r + 1) / 2` |
| 溢出 | `(l + r) / 2` → `l + (r - l) / 2` |
| 单调性反 | 画 `check(mid)` 真值表 |
| 边界差 1 | 写**单元素 / 两元素**测试 |

---

## 总结

```text
找数 → 模板 1
找位置 → 模板 2 / 3
最优解 + 单调可判定 → 二分答案
```

源条目：
- [[../../wiki/algorithms/binary-search]]
- [[../../wiki/techniques/binary-search-answer]]
- [[../../wiki/maps/algorithm-overview]]
