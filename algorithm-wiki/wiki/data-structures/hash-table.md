---
title: 哈希表
aliases: [Hash Table, Hash Map, 散列表, unordered_map, unordered_set]
tags: [data-structure, hash]
created: 2026-05-03
updated: 2026-05-03
---

# 哈希表

> **TL;DR**：哈希表 = **哈希函数 + 桶**。把 key 通过 hash 映射到数组下标，期望 O(1) 完成 insert / find / delete。是面试**最高频**的数据结构之一。

## C++ 容器选型

| 容器 | 底层 | 操作复杂度 | 何时用 |
|---|---|---|---|
| `unordered_map<K, V>` | 哈希表 | 平均 O(1)，最坏 O(N) | 一般情况 |
| `unordered_set<K>` | 哈希表 | 平均 O(1) | 只关心 key |
| `map<K, V>` | 红黑树 | O(log N) | 需要**有序遍历** |
| `set<K>` | 红黑树 | O(log N) | 需要 lower_bound / 有序 |

**面试默认**：`unordered_map` / `unordered_set`。**算法竞赛**：偶尔用 `map` 防 hack，因为 unordered 在恶意构造的输入下能被打到 O(N²)。

## 用法速查

```cpp
unordered_map<string, int> mp;
mp["a"] = 1;                       // 插入或更新
mp.insert({"b", 2});               // 插入
if (mp.count("a")) ...;            // 是否存在
if (auto it = mp.find("a"); it != mp.end()) ...;   // 找到值
mp.erase("a");                     // 删除
for (auto& [k, v] : mp) ...;       // 遍历

// 计数器
unordered_map<int, int> cnt;
for (int x : nums) cnt[x]++;       // operator[] 不存在则创建默认值（int 默认 0）

// 字符串 hash key
unordered_map<string, vector<string>> groups;   // 字母异位词分组
```

## 哈希冲突的两种处理

### 1. 拉链法（Chaining）

每个桶是一个**链表**（或小型数组）：

```text
桶 0: [k1] -> [k7]
桶 1: [k2]
桶 2: (空)
桶 3: [k4] -> [k8] -> [k11]
```

C++ `unordered_map` 用拉链。

### 2. 开放寻址（Open Addressing）

只用一个大数组，冲突时按规则探测下一个空位：线性探测、二次探测、双重哈希。

Python `dict`、Go `map`、Rust `HashMap` 都用开放寻址（更 cache-friendly）。

## 装载因子 & 扩容

- **装载因子 α = 元素数 / 桶数**
- α 太大 → 冲突频繁 → 退化成 O(N)
- 触发**rehash**：桶数翻倍，把所有元素重新映射

C++ `unordered_map` 默认 α ≤ 1.0 触发 rehash。可以预估大小先 `reserve`：

```cpp
unordered_map<int, int> mp;
mp.reserve(100000);    // 减少 rehash 次数
```

## 哈希函数

C++ STL 内置 `std::hash`：

```cpp
size_t h1 = hash<int>{}(42);
size_t h2 = hash<string>{}("hello");
```

**自定义类型作为 key**，要提供 hash + equality：

```cpp
struct Point { int x, y; };

struct PointHash {
    size_t operator()(const Point& p) const {
        return hash<int>{}(p.x) ^ (hash<int>{}(p.y) << 1);
    }
};

struct PointEq {
    bool operator()(const Point& a, const Point& b) const {
        return a.x == b.x && a.y == b.y;
    }
};

unordered_map<Point, int, PointHash, PointEq> mp;
```

或者更简单——用 `pair<int,int>` 配 `boost::hash_combine`，或者直接编码成 `long long key = (long long)x * MOD + y`。

## 何时想到哈希表

太多了，最高频几类：

| 信号词 | 例子 |
|---|---|
| "两数之和 / k-sum" | 边遍历边查互补数 |
| "去重" | 用 set 看是否见过 |
| "频次统计" | map<元素, 计数> |
| "存某个映射" | 字符 → 索引、key → 上次出现位置 |
| "anagram / 字母异位词" | 字符计数当哈希 key |
| "[[problems/lru\|LRU]] / LFU" | hash + 双链表 |
| "前缀和 + 目标" | 见 [[prefix-sum]] |

## 经典题型

| 题 | 用法 |
|---|---|
| 两数之和 LC 1 | 边遍历边查 `target - x` |
| 字母异位词分组 LC 49 | 排序后字符串当 key |
| 最长无重复字符串 LC 3 | hash 记上次出现位置 |
| 和为 K 的子数组 LC 560 | 前缀和 + hash |
| LRU 缓存 LC 146 | hash + 双链表，见 [[problems/lru]] |
| 同构字符串 LC 205 | 双向映射 |
| 第一个不重复的字符 LC 387 | 计数 |
| 找所有数组中消失的数字 LC 448 | 用数组当 hash（值域 0~N） |

## 性能对比

实测中：

- `unordered_map` 平均比 `map` 快 2~5 倍
- 但**冲突恶化**时可能比 map 慢
- **小规模**（< 100 元素）`map` 反而更快（缓存友好、常数小）

## 易错点

- 用 `mp[key]` 访问会**自动创建**默认值。判断存在用 `mp.count(key)` 或 `mp.find()`
- `unordered_map` **遍历顺序不确定**，依赖顺序的代码会出 bug
- 自定义 key 必须提供 hash + equality
- 哈希被恶意构造数据 hack 时考虑加随机种子或换 `map`
- 大量 insert 前先 `reserve` 提速

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「哈希表」相关关键词（哈希, hash）的笔记，按来源分组。

### 灵茶山艾府和代码随想录
- [[哈希表|课程/灵茶山艾府和代码随想录/哈希表.md]]

### 左程云
- [[哈希表增删改查在使用时可以认为是O(1)|课程/左程云/001. 难点重点照顾/哈希表增删改查在使用时可以认为是O(1).md]]
- [[Java Object 之hashcode|课程/左程云/002. 知识点总结/00. 编程语言相关/Java Object 之hashcode.md]]
- [[Java中HashMap, TreeMap 等的元素遍历顺序|课程/左程云/002. 知识点总结/00. 编程语言相关/Java中HashMap, TreeMap 等的元素遍历顺序.md]]
- [[Java中HashSet vs TreeSet|课程/左程云/002. 知识点总结/00. 编程语言相关/Java中HashSet vs TreeSet.md]]
- [[Java中的哈希表|课程/左程云/002. 知识点总结/00. 编程语言相关/Java中的哈希表.md]]
- [[一致性哈希|课程/左程云/002. 知识点总结/20. 算法知识点/一致性哈希.md]]

## 参考资料

- [[../../课程/灵茶山艾府和代码随想录/哈希表]] — 你整理的哈希专题
- [[../../语言/C++/STL]] — STL 容器用法

## 关联条目

- 上层：[[ds-overview]]
- 兄弟：[[trie]]（也是 key→val 映射的一种）
- 应用：[[problems/lru]] · [[two-pointers]] · [[sliding-window]] · [[prefix-sum]]
