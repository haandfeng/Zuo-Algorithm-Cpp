---
title: 链表
aliases: [Linked List, 单链表, 双链表, 哨兵]
tags: [data-structure, linear]
created: 2026-05-03
updated: 2026-05-03
---

# 链表

> **TL;DR**：链表 = **节点 + 指针**串成的线性结构。优势是**任意位置插删 O(1)**（已持有指针），代价是**不能随机访问 + cache 不友好**。

## 节点定义

```cpp
struct ListNode {
    int val;
    ListNode* next;
    ListNode(int x = 0, ListNode* n = nullptr) : val(x), next(n) {}
};

// 双链表
struct DListNode {
    int val;
    DListNode *prev, *next;
};
```

## 哨兵（dummy head）

链表题最容易出 bug 的位置是**头节点处理**——头被删 / 头是 null。万能解法：**在真头前加一个虚节点**：

```cpp
ListNode dummy(0, head);            // 不在堆上
ListNode* p = &dummy;
while (p->next) {
    if (p->next->val == val) p->next = p->next->next;
    else p = p->next;
}
return dummy.next;
```

这样**所有节点都有前驱**，删除 / 插入只用一种代码路径。

## 复杂度速查

| 操作 | 单链表 | 双链表 |
|---|---|---|
| 已持指针位置插入 / 删除 | O(1) | O(1) |
| 头插 / 头删 | O(1) | O(1) |
| 尾插（无尾指针） | O(N) | O(1)（有尾指针） |
| 按值查找 | O(N) | O(N) |
| 按下标访问 | O(N) | O(N) |

## 三种核心操作模板

### 1. 反转

迭代版（最常考）：

```cpp
ListNode* reverse(ListNode* head) {
    ListNode *prev = nullptr, *cur = head;
    while (cur) {
        ListNode* next = cur->next;
        cur->next = prev;
        prev = cur;
        cur = next;
    }
    return prev;
}
```

递归版：

```cpp
ListNode* reverse(ListNode* head) {
    if (!head || !head->next) return head;
    ListNode* newHead = reverse(head->next);
    head->next->next = head;
    head->next = nullptr;
    return newHead;
}
```

详见 [[problems/reverse-linked-list]]。

### 2. 快慢指针

**找中点**：

```cpp
ListNode* middle(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
    }
    return slow;     // 偶数长度返回靠右那个中点
}
```

**判环**（Floyd 判圈）：

```cpp
bool hasCycle(ListNode* head) {
    ListNode *slow = head, *fast = head;
    while (fast && fast->next) {
        slow = slow->next;
        fast = fast->next->next;
        if (slow == fast) return true;
    }
    return false;
}
```

**找环入口**：相遇后让一个指针回到 head，两个一起一步步走，再次相遇就是入口。

### 3. 合并两个有序链表

```cpp
ListNode* merge(ListNode* a, ListNode* b) {
    ListNode dummy;
    ListNode* tail = &dummy;
    while (a && b) {
        if (a->val <= b->val) { tail->next = a; a = a->next; }
        else                   { tail->next = b; b = b->next; }
        tail = tail->next;
    }
    tail->next = a ? a : b;
    return dummy.next;
}
```

是 [[sorting|归并排序]] 链表版的核心。

## 链表 vs 数组

| 维度 | 数组 | 链表 |
|---|---|---|
| 访问 | O(1) | O(N) |
| 中间插删 | O(N) | O(1)（已持指针） |
| 内存连续 | ✅ → cache 友好 | ❌ |
| 内存利用率 | 100% | 多用 8 字节存 next |
| 大小固定 | 否（vector 可变） | 否 |
| 实现复杂度 | 简单 | 中等（指针易错） |

**经验**：90% 的工程场景**用 vector 就够**。链表主要在算法题、操作系统内核、特定数据库索引里出现。

## 经典题型

| 题 | 技巧 |
|---|---|
| 反转链表 LC 206 | 迭代 / 递归 |
| K 个一组反转 LC 25 | 头尾指针 + 反转子段 |
| 环形链表 LC 141 / 142 | 快慢指针 |
| 合并 K 个有序链表 LC 23 | 优先队列 |
| LRU LC 146 | [[linked-list\|双链表]] + [[hash-table\|哈希]]，见 [[problems/lru]] |
| 复制带随机指针 LC 138 | 哈希 / 原地穿插 |
| 回文链表 LC 234 | 找中点 + 反转后半 |
| 重排链表 LC 143 | 找中点 + 反转 + 合并 |

## 易错点

- 忘记用 dummy 处理头节点
- 修改 `cur->next` 之前没保存原 next → 链断了
- 用 `cur != nullptr` vs `cur->next != nullptr`：差一步会 segfault
- 双链表插入删除要更新**两个**方向的指针

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「链表」相关关键词（链表）的笔记，按来源分组。

### 灵茶山艾府和代码随想录
- [[链表|课程/灵茶山艾府和代码随想录/链表.md]]

### 左程云
- [[两个可能有环的单链表相交的第一个节点-难点释疑|课程/左程云/001. 难点重点照顾/01. 难点释疑/两个可能有环的单链表相交的第一个节点-难点释疑.md]]
- [[链表相关问题|课程/左程云/001. 难点重点照顾/链表相关问题.md]]
- [[两个无环链表返回第一个相交节点|课程/左程云/003. 编程基础技巧及工具/两个无环链表返回第一个相交节点.md]]
- [[两个有环链表返回第一个相交节点|课程/左程云/003. 编程基础技巧及工具/两个有环链表返回第一个相交节点.md]]
- [[利用快慢指针返回链表的中点|课程/左程云/003. 编程基础技巧及工具/利用快慢指针返回链表的中点.md]]
- [[判断链表是否有环|课程/左程云/010. 代码模板/判断链表是否有环.md]]

### labuladong
- [[双指针-链表 和 链表代码实现|课程/labuladong/核心刷题框架/双指针-链表 和 链表代码实现.md]]

### C++算法
- [[链表高频题和必备技巧|课程/C++算法/必备/链表高频题和必备技巧.md]]

### 题目
- [[0019-删除链表的倒数第 N 个结点|题目/0019-删除链表的倒数第 N 个结点.md]]
- [[0021-合并两个有序链表|题目/0021-合并两个有序链表.md]]
- [[0023-合并 K 个升序链表|题目/0023-合并 K 个升序链表.md]]
- [[0024-两两交换链表中的节点|题目/0024-两两交换链表中的节点.md]]
- [[0025-K 个一组翻转链表|题目/0025-K 个一组翻转链表.md]]
- [[0061-旋转链表|题目/0061-旋转链表.md]]

## 参考资料

- [[../../课程/灵茶山艾府和代码随想录/链表]] — 你整理的链表专题
- [[../../课程/C++算法/必备/链表高频题]] — C++ 算法笔记里的链表（如有）
- [[../../课程/labuladong/核心刷题框架]] — labuladong 的链表框架

## 关联条目

- 上层：[[ds-overview]]
- 兄弟：[[array]]
- 应用：[[problems/reverse-linked-list]] · [[problems/lru]]
- 技巧：[[two-pointers]]
