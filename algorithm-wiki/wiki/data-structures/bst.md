---
title: 二叉搜索树
aliases: [BST, Binary Search Tree, 二叉排序树]
tags: [data-structure, tree]
created: 2026-05-03
updated: 2026-05-03
---

# 二叉搜索树（BST）

> **TL;DR**：BST 是一种特殊的 [[binary-tree|二叉树]]：**任一节点的左子树所有值 < 它 < 右子树所有值**。中序遍历得**升序序列**。是有序集合 / 有序映射的实现基础。

## 核心性质

```text
       5
      / \
     3   7
    / \   \
   1   4   8
```

- **左子树所有值 < 当前节点 < 右子树所有值**（严格不等）
- **中序遍历**得升序序列：1, 3, 4, 5, 7, 8
- **没有重复值**（重复用 multiset 或额外 count）

## 三个核心操作

```cpp
// 1. 查找
TreeNode* search(TreeNode* root, int target) {
    if (!root || root->val == target) return root;
    return target < root->val ? search(root->left, target)
                              : search(root->right, target);
}

// 2. 插入
TreeNode* insert(TreeNode* root, int val) {
    if (!root) return new TreeNode(val);
    if (val < root->val) root->left = insert(root->left, val);
    else if (val > root->val) root->right = insert(root->right, val);
    return root;
}

// 3. 删除（最难）
TreeNode* deleteNode(TreeNode* root, int key) {
    if (!root) return nullptr;
    if (key < root->val) root->left = deleteNode(root->left, key);
    else if (key > root->val) root->right = deleteNode(root->right, key);
    else {
        // 找到要删的节点
        if (!root->left) return root->right;
        if (!root->right) return root->left;
        // 两个孩子都在：用右子树最小节点替换当前节点
        TreeNode* succ = root->right;
        while (succ->left) succ = succ->left;
        root->val = succ->val;
        root->right = deleteNode(root->right, succ->val);
    }
    return root;
}
```

## 复杂度

| 操作 | 平衡（log N） | 退化成链表（最坏 N） |
|---|---|---|
| 查找 | O(log N) | O(N) |
| 插入 | O(log N) | O(N) |
| 删除 | O(log N) | O(N) |
| 中序遍历 | O(N) | O(N) |

**朴素 BST 不保证平衡**——最坏（顺序插入）退化成链表。所以工程实现都用平衡 BST。

## 平衡 BST 家族

| 类型 | 平衡条件 | 代表实现 |
|---|---|---|
| **AVL 树** | 任一节点左右子树高度差 ≤ 1 | 严格平衡，查找快但插删慢 |
| **红黑树** | 5 条颜色性质（约 2 倍 log N） | C++ `map`/`set`、Java `TreeMap`、Linux 内核 |
| **Treap** | BST 性质 + 随机优先级满足堆性质 | 算法竞赛常用 |
| **Splay 树** | 每次访问把节点旋转到根 | 工程少用 |
| **跳表（Skip List）** | 不是 BST，但同等功能 | Redis ZSET |

C++ 的 `std::map` 和 `std::set` 都用**红黑树**。

## C++ 中的"BST"使用

```cpp
set<int> s;
s.insert(5);
s.insert(3);
s.insert(7);

// 有序遍历
for (int x : s) cout << x << " ";   // 3 5 7

// 找 ≥ x 的最小值（lower_bound）
auto it = s.lower_bound(4);    // 指向 5

// 找 > x 的最小值（upper_bound）
it = s.upper_bound(5);         // 指向 7

// 找前驱（< x 的最大值）
it = s.lower_bound(5);
if (it != s.begin()) --it;    // 指向 3

// 删除
s.erase(5);
```

`map<K, V>` 用法类似，但是 key→value 的有序映射。

## 何时想到 BST / set / map

| 信号 | 例子 |
|---|---|
| "需要有序" | 有序去重、按序遍历 |
| "找前驱 / 后继" | lower_bound / upper_bound |
| "动态求第 K 大" | multiset / 平衡 BST |
| "区间查询有序集" | 推荐 [[segment-tree\|线段树]] 或树状数组 |
| "按时间窗口维护" | set + 滑窗 |

## 经典题型

| 题 | 用法 |
|---|---|
| 验证 BST LC 98 | 中序 / 上下界 |
| BST 第 K 小 LC 230 | 中序 + 计数 |
| BST 转有序双链表 LC 426 | 中序 |
| BST 删除节点 LC 450 | 上方模板 |
| 二叉搜索树迭代器 LC 173 | 中序栈式迭代 |
| 数据流的中位数 LC 295 | 双堆（不是 BST 但同思路） |
| 滑动窗口中位数 LC 480 | multiset / 双堆 |
| 我的日程安排表 II LC 731 | set + lower_bound |
| 区间和的个数 LC 327 | 树状数组 / 归并 |

## 易错点

- **不是平衡** → 朴素 BST 在排序输入上退化
- **删除两个孩子的节点** → 必须找前驱或后继替换
- **重复值** → 朴素 BST 不允许重复，要用 multiset 或附加 count 字段
- **中序遍历是升序**这个性质要会主动用，比如 LC 98 验证 BST 用中序最直观

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「BST」相关关键词（二叉搜索, BST, 搜索树）的笔记，按来源分组。

### 左程云
- [[不同的二叉搜索树|课程/左程云/200. ★算法题目汇总★/★算法面试题汇总/不同的二叉搜索树.md]]
- [[二叉搜索子树的最大键值和|课程/左程云/200. ★算法题目汇总★/★算法面试题汇总/二叉搜索子树的最大键值和.md]]
- [[二叉搜索树中的顺序后继|课程/左程云/200. ★算法题目汇总★/★算法面试题汇总/二叉搜索树中的顺序后继.md]]
- [[二叉搜索树中第K小的元素|课程/左程云/200. ★算法题目汇总★/★算法面试题汇总/二叉搜索树中第K小的元素.md]]
- [[二叉树中最大的二叉搜索子树的大小|课程/左程云/200. ★算法题目汇总★/★算法面试题汇总/二叉树中最大的二叉搜索子树的大小.md]]
- [[二叉树中最大的二叉搜索子树的头节点|课程/左程云/200. ★算法题目汇总★/★算法面试题汇总/二叉树中最大的二叉搜索子树的头节点.md]]

## 参考资料

- [[../../课程/C++算法/必备/二叉树经典问题]] — 二叉树章节包含 BST
- [[../../语言/C++/STL]] — set / map 用法
- 《算法导论》12 / 13 章

## 关联条目

- 上层：[[binary-tree]] · [[ds-overview]]
- 兄弟：[[heap]]
- 衍生：[[segment-tree]] · [[fenwick-tree]]
