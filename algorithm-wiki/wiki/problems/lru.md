---
title: LRU 缓存
aliases: [LRU, LC 146, Least Recently Used]
tags: [problem, design, hash, linked-list]
difficulty: medium
leetcode: 146
created: 2026-05-03
updated: 2026-05-03
---

# LRU 缓存（LC 146）

> **题号**：LeetCode 146
> **难度**：Medium（设计题）
> **核心**：**[[../data-structures/hash-table|哈希表]] + [[../data-structures/linked-list|双向链表]]** 的经典组合，让 `get` / `put` 都达到 **O(1)**。

## 题目

设计 LRU（Least Recently Used）缓存，支持：

- `get(key)`：存在则返回值并把它**变成最近使用**；否则返回 -1
- `put(key, value)`：插入或更新键值对；若超出容量，**淘汰最久未使用**的键

要求 `get` 和 `put` 都 **O(1)**。

## 设计思路

为什么 `O(1)` 难做：

- **快速查找** → 哈希表
- **快速调整使用顺序** → 双向链表（O(1) 移动节点）
- **关联起来** → 哈希表的 value 直接存**链表节点指针**

```text
       hash:  key → DListNode*
           ↓
    head ↔ N1 ↔ N2 ↔ N3 ↔ tail
    (最新)               (最旧)
```

每次 `get` / `put` 把节点移到 head 后；容量满时把 tail 前的节点删掉。

## 实现

```cpp
class LRUCache {
    struct Node {
        int key, val;
        Node *prev, *next;
        Node(int k = 0, int v = 0) : key(k), val(v), prev(nullptr), next(nullptr) {}
    };
    int cap;
    unordered_map<int, Node*> mp;
    Node *head, *tail;        // dummy nodes

    void remove(Node* node) {
        node->prev->next = node->next;
        node->next->prev = node->prev;
    }
    void addToFront(Node* node) {
        node->next = head->next;
        node->prev = head;
        head->next->prev = node;
        head->next = node;
    }

public:
    LRUCache(int capacity) : cap(capacity) {
        head = new Node();
        tail = new Node();
        head->next = tail;
        tail->prev = head;
    }

    int get(int key) {
        if (!mp.count(key)) return -1;
        Node* node = mp[key];
        remove(node);
        addToFront(node);
        return node->val;
    }

    void put(int key, int value) {
        if (mp.count(key)) {
            Node* node = mp[key];
            node->val = value;
            remove(node);
            addToFront(node);
        } else {
            if ((int)mp.size() == cap) {
                Node* old = tail->prev;
                remove(old);
                mp.erase(old->key);
                delete old;
            }
            Node* node = new Node(key, value);
            mp[key] = node;
            addToFront(node);
        }
    }
};
```

时间 O(1)、空间 O(capacity)。

## 关键设计点

### 1. 为什么用 dummy head + tail

避免每次操作都判空——边界节点处理永远统一。

### 2. 为什么节点要存 key

淘汰末尾节点时**要从哈希表里删除对应的 key**——所以节点必须知道自己的 key。

### 3. 为什么是双向链表

要 O(1) 删除节点 → 必须知道**前驱**。单链表只能 O(N) 找前驱。

### 4. 也可以用 std::list

C++ STL 的 `std::list<pair<int,int>>` 是双向链表，配合 `unordered_map<int, list<pair<int,int>>::iterator>` 同样能 O(1)：

```cpp
class LRUCache {
    int cap;
    list<pair<int,int>> lst;                    // (key, value)
    unordered_map<int, list<pair<int,int>>::iterator> mp;
public:
    LRUCache(int capacity) : cap(capacity) {}

    int get(int key) {
        auto it = mp.find(key);
        if (it == mp.end()) return -1;
        lst.splice(lst.begin(), lst, it->second);    // O(1) 移到头
        return it->second->second;
    }

    void put(int key, int value) {
        auto it = mp.find(key);
        if (it != mp.end()) {
            it->second->second = value;
            lst.splice(lst.begin(), lst, it->second);
            return;
        }
        if ((int)mp.size() == cap) {
            mp.erase(lst.back().first);
            lst.pop_back();
        }
        lst.emplace_front(key, value);
        mp[key] = lst.begin();
    }
};
```

`splice` 是 O(1)——移动节点而不拷贝。代码短得多，**面试推荐**。

## 变体

| 题 | 变化 |
|---|---|
| LFU 缓存 LC 460 | Least Frequently Used，需要按频次组织 |
| 设计跳表 LC 1206 | 完全不同的数据结构 |
| 全 O(1) 数据结构 LC 432 | inc / dec / max / min 都 O(1) |
| LRU + TTL | 加时间戳 + 定时清理 |

## 易错点

- 用单链表 → 删除节点要 O(N) 找前驱
- 用 `vector` / `deque` → 中间删除是 O(N)
- 节点没存 key → 淘汰时哈希表删不掉
- `addToFront` 和 `remove` 顺序写反 → 链断
- 容量为 0 的边界

## 在你 Vault 中的位置

- [[../../Hot100|Hot100]] 必考
- [[../../Blind75|Blind75]] 必考
- [[../../labuladong/经典数据结构算法]] — labuladong 单独讲过

## 关联条目

- 数据结构：[[../data-structures/linked-list]] · [[../data-structures/hash-table]]
- 兄弟题：[[reverse-linked-list]]
- 设计题：（属于 LeetCode 设计大类）
