---
title: 反转链表
aliases: [Reverse Linked List, LC 206]
tags: [problem, linked-list, classic]
difficulty: easy
leetcode: 206
created: 2026-05-03
updated: 2026-05-03
---

# 反转链表（LC 206）

> **题号**：LeetCode 206 / 剑指 Offer 24
> **难度**：Easy（但变体满天飞）
> **核心**：链表第一题，**必须能闭着眼写出迭代和递归两种**。

## 题目

给定单链表的头节点 `head`，反转链表，并返回反转后的头节点。

```text
输入: 1 → 2 → 3 → 4 → 5
输出: 5 → 4 → 3 → 2 → 1
```

## 解法 1：迭代（推荐）

```cpp
ListNode* reverseList(ListNode* head) {
    ListNode *prev = nullptr, *cur = head;
    while (cur) {
        ListNode* next = cur->next;     // 1. 先存下下一个
        cur->next = prev;                // 2. 反转当前指针
        prev = cur;                      // 3. prev 前进
        cur = next;                      // 4. cur 前进
    }
    return prev;
}
```

时间 O(N)、空间 O(1)。**4 步顺序不能换**。

直观图示：

```text
初始:   null  ←  1 → 2 → 3
                prev  cur

第 1 轮:  null ← 1   ←  2 → 3
                  prev  cur

第 2 轮:  null ← 1 ← 2   ←  3
                       prev  cur

最后:    null ← 1 ← 2 ← 3
                          prev
```

## 解法 2：递归

```cpp
ListNode* reverseList(ListNode* head) {
    if (!head || !head->next) return head;
    ListNode* newHead = reverseList(head->next);
    head->next->next = head;
    head->next = nullptr;
    return newHead;
}
```

时间 O(N)、空间 O(N)（递归栈深 = N）。

**直觉**：

- `reverseList(head->next)` 已经把 `head->next` 之后的部分反转，`newHead` 是新的头
- 此时原本 `head → head->next → (反转后部分末尾)` 变成 `head → head->next → (反转后)` 中 `head->next` 是反转后的尾
- 所以 `head->next->next = head` 把 head 接到尾巴上
- `head->next = nullptr` 让 head 成为新的尾

## 变体题

| 题 | 变化点 |
|---|---|
| 反转链表 II LC 92 | 反转 `[m, n]` 区间内 |
| K 个一组翻转链表 LC 25 | 每 K 个翻一组 |
| 两两交换链表中的节点 LC 24 | K=2 的特例 |
| 旋转链表 LC 61 | 整体右移 K 位 |
| 回文链表 LC 234 | 找中点 + 反转后半 + 比较 |
| 重排链表 LC 143 | 找中点 + 反转后半 + 合并 |

### 变体：反转区间 [m, n]

```cpp
ListNode* reverseBetween(ListNode* head, int m, int n) {
    ListNode dummy(0, head);
    ListNode* prev_m = &dummy;
    for (int i = 1; i < m; ++i) prev_m = prev_m->next;
    ListNode* cur = prev_m->next;
    for (int i = 0; i < n - m; ++i) {
        ListNode* next = cur->next;
        cur->next = next->next;
        next->next = prev_m->next;
        prev_m->next = next;
    }
    return dummy.next;
}
```

**头插法**：把 `cur->next` 反复插到 `prev_m` 之后。

### 变体：K 个一组（LC 25）

```cpp
ListNode* reverseKGroup(ListNode* head, int k) {
    ListNode dummy(0, head);
    ListNode* prev_group = &dummy;
    while (true) {
        // 检查剩余是否还有 k 个
        ListNode* tail = prev_group;
        for (int i = 0; i < k; ++i) {
            tail = tail->next;
            if (!tail) return dummy.next;
        }
        // 反转 [prev_group->next, tail]
        ListNode* group_head = prev_group->next;
        ListNode* next_group = tail->next;
        // 标准反转
        ListNode *prev = next_group, *cur = group_head;
        while (cur != next_group) {
            ListNode* next = cur->next;
            cur->next = prev;
            prev = cur;
            cur = next;
        }
        prev_group->next = tail;
        prev_group = group_head;
    }
}
```

## 易错点

- 迭代版：忘记**先存 next 再改指针** → 链断了
- 递归版：忘 `head->next = nullptr` → 出现环
- 边界：`head` 为 null 或只有 1 个节点时直接返回
- LC 92 区间反转：m=1 时要用 dummy 处理，否则 `prev_m` 不存在

## 与其它概念的关系

- 是 [[linked-list]] 章节的"基本功"题
- 是 [[recursion|递归思维]] 的入门练习
- LC 25 反转 K 一组用到了"先反转一段再连接"的模板，是工程题常考

## 参考资料

- [[../../linked-list]]（在本 wiki 内：[[../data-structures/linked-list]]）
- [[../../零茶山艾府+代码随想录/链表]]
- [[../../labuladong/核心刷题框架]] — labuladong 反转链表

## 关联条目

- 数据结构：[[../data-structures/linked-list]]
- 思维：[[../concepts/recursion]]
- 兄弟题：[[lru]]
