---
title: 树形 DP
aliases: [Tree DP]
tags: [pattern, dp, tree]
created: 2026-05-03
updated: 2026-05-03
---

# 树形 DP

> **TL;DR**：在树上做 DP——**状态以子树为单位**，用一次 [[../algorithms/bfs-dfs|DFS]] 后序遍历自底向上计算。是 [[../data-structures/binary-tree|二叉树]] 题的最高频范式。

## 核心套路

```cpp
// 返回：以 root 为根的子树的"某统计量"
SomeType dfs(TreeNode* root) {
    if (!root) return base_value;
    auto left  = dfs(root->left);
    auto right = dfs(root->right);
    // 综合 left, right, root->val 算出当前子树的答案
    // 必要时更新全局 ans
    return value_for_my_subtree;
}
```

**核心**：[[../concepts/recursion#模板：左程云"递归三问"|递归三问]]——
1. 函数返回什么（语义）
2. 边界（空节点）
3. 由左右子树结果如何推父节点

## 三类经典

### 1. 选 / 不选

经典：打家劫舍 III（LC 337）

```cpp
// 返回 {不偷当前节点, 偷当前节点}
pair<int,int> dfs(TreeNode* root) {
    if (!root) return {0, 0};
    auto [ln, lr] = dfs(root->left);
    auto [rn, rr] = dfs(root->right);
    int notRob = max(ln, lr) + max(rn, rr);
    int rob    = root->val + ln + rn;
    return {notRob, rob};
}
```

### 2. 子树最优 + 全局答案

经典：树的直径（LC 543）、二叉树的最大路径和（LC 124）

```cpp
int ans = INT_MIN;
int dfs(TreeNode* root) {
    if (!root) return 0;
    int l = max(0, dfs(root->left));      // 负贡献剪掉
    int r = max(0, dfs(root->right));
    ans = max(ans, root->val + l + r);    // 当前节点为最高点的路径
    return root->val + max(l, r);          // 只能延伸一边
}
```

### 3. 换根 DP

求"以每个节点为根时的某统计量"。第一遍 DFS 算以节点 0 为根的答案，第二遍 DFS **基于父节点的答案推子节点**。

经典：树中距离之和（LC 834）。

## 何时想到树形 DP

| 信号 | 例子 |
|---|---|
| "求一棵树上的最大 / 最小 / 计数" | 树形 DP |
| "选 / 不选某个节点" | 打家劫舍 III、最大独立集 |
| "求每个节点为根时的答案" | 换根 DP |
| "树上路径相关" | LC 124 / 543 |

## 经典题型

| 题 | 类型 |
|---|---|
| 二叉树的直径 LC 543 | 子树最优 + 全局 |
| 二叉树的最大路径和 LC 124 | 同上 |
| 打家劫舍 III LC 337 | 选 / 不选 |
| 监控二叉树 LC 968 | 三态 DP |
| 没有上司的舞会 | 选 / 不选（多叉） |
| 树的重心 | 一遍 DFS |
| 树中距离之和 LC 834 | 换根 |

## 易错点

- 子树值**有正有负** → 用 `max(0, dfs(child))` 剪掉负贡献
- 全局变量 `ans` 要重置 / 初值正确（INT_MIN 还是 0？）
- 返回的"贡献"和"答案"是两个不同的量——容易混
- 多叉树的转移要用 for 循环遍历所有孩子
- 递归过深爆栈：用迭代或开大栈空间

## 参考资料

- [[../../labuladong/经典数据结构算法]] — labuladong 树形 DP
- [[../../算法笔记/zuoAlgorithm/100. 算法课程]] — 左程云树形 DP 章节

## 关联条目

- 上层：[[dp]] · [[dp-roadmap]]
- 数据结构：[[../data-structures/binary-tree]]
- 思维：[[../concepts/recursion]]
- 兄弟：[[dp-knapsack]] · [[dp-interval]] · [[dp-bitmask]]
