---
title: 二叉树
aliases: [Binary Tree, BT, 二叉树遍历]
tags: [data-structure, tree]
created: 2026-05-03
updated: 2026-05-03
---

# 二叉树

> **TL;DR**：二叉树 = 每个节点最多 2 个子节点的树。是 LeetCode 题量最大的数据结构之一。**学好递归就学好了二叉树的一半**。

## 节点定义

```cpp
struct TreeNode {
    int val;
    TreeNode *left, *right;
    TreeNode(int x = 0, TreeNode* l = nullptr, TreeNode* r = nullptr)
        : val(x), left(l), right(r) {}
};
```

## 4 种基本遍历

### 1. 前序（Preorder：根 → 左 → 右）

```cpp
void preorder(TreeNode* root, vector<int>& res) {
    if (!root) return;
    res.push_back(root->val);
    preorder(root->left, res);
    preorder(root->right, res);
}
```

### 2. 中序（Inorder：左 → 根 → 右）

```cpp
void inorder(TreeNode* root, vector<int>& res) {
    if (!root) return;
    inorder(root->left, res);
    res.push_back(root->val);
    inorder(root->right, res);
}
```

**对 [[bst|BST]] 中序遍历得到升序序列**——这是 BST 的"杀手特性"。

### 3. 后序（Postorder：左 → 右 → 根）

```cpp
void postorder(TreeNode* root, vector<int>& res) {
    if (!root) return;
    postorder(root->left, res);
    postorder(root->right, res);
    res.push_back(root->val);
}
```

后序适合**自底向上汇总信息**（比如最大深度、直径、最近公共祖先）。

### 4. 层序（BFS：一层一层）

```cpp
vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> res;
    if (!root) return res;
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int sz = q.size();
        vector<int> level;
        for (int i = 0; i < sz; ++i) {
            auto* node = q.front(); q.pop();
            level.push_back(node->val);
            if (node->left)  q.push(node->left);
            if (node->right) q.push(node->right);
        }
        res.push_back(level);
    }
    return res;
}
```

详见 [[bfs-dfs]]。

## 三种思考方式（左程云风格）

针对二叉树问题，左程云课程总结了**3 种通用套路**：

### 套路 1：DFS 遍历传参

"我作为某个节点，能从父节点拿到什么、传给孩子什么"。

```cpp
void dfs(TreeNode* root, int depth, vector<int>& res) {
    if (!root) return;
    if (depth == res.size()) res.push_back(root->val);   // 第一个被访问到的节点
    dfs(root->left, depth + 1, res);
    dfs(root->right, depth + 1, res);
}
```

### 套路 2：DFS 收集子树信息（"递归三问"）

"我从左右孩子分别拿到 X、Y，要返回什么给父节点"。

```cpp
// 最大深度
int maxDepth(TreeNode* root) {
    if (!root) return 0;
    return max(maxDepth(root->left), maxDepth(root->right)) + 1;
}

// 直径（树形 DP）
int diameter = 0;
int depth(TreeNode* root) {
    if (!root) return 0;
    int l = depth(root->left);
    int r = depth(root->right);
    diameter = max(diameter, l + r);
    return max(l, r) + 1;
}
```

详见 [[recursion#模板：左程云"递归三问"]]。

### 套路 3：BFS 层序

适合**和层数有关**的题：右视图、最大宽度、Z 字形遍历。

## 经典性质

| 性质 | 说明 |
|---|---|
| 节点数 | N |
| 边数 | N - 1 |
| 高度（递归定义） | 空树 -1，叶子 0，否则 max(左, 右) + 1 |
| 完全二叉树第 k 层节点数 | 最多 2^k |
| 高度 h 完全二叉树节点数 | 2^(h+1) - 1 |
| BFS 第 k 层节点数 | 第 k 层有几个 |
| 由前序+中序唯一确定一棵树 | ✅ |
| 由前序+后序**不能**唯一确定 | ❌（缺中间分界） |

## 经典题型（高频）

| 题 | 套路 |
|---|---|
| 最大深度 LC 104 | 套路 2 |
| 直径 LC 543 | 套路 2 + 全局变量 |
| 平衡二叉树 LC 110 | 套路 2 |
| 对称二叉树 LC 101 | 套路 2（双递归） |
| 翻转二叉树 LC 226 | 套路 2 |
| 验证 BST LC 98 | 中序 / 上下界 |
| 二叉树展开为链表 LC 114 | 后序 |
| 路径总和 III LC 437 | 前缀和 + DFS（你已写） |
| 最近公共祖先 LC 236 | 后序 |
| 序列化 / 反序列化 LC 297 | 前序（含 null 标记） |
| 从前序+中序构造 LC 105 | 分治 |
| 二叉树的最大路径和 LC 124 | 树形 DP |

## 完全二叉树（特殊形）

完全二叉树 = 除了最后一层都满 + 最后一层左对齐。

性质：

- 可以用**数组**存：父 = `i / 2`，左孩子 = `2i`，右孩子 = `2i + 1`（下标从 1 开始）
- 节点编号 1 ~ N
- **高度 = ⌊log₂ N⌋**
- **[[heap|堆]]** 就是完全二叉树的数组表示

## 易错点

- **递归时机**：先访问根再递归 vs 先递归再访问根
- **base case**：空节点返回 0 / -1 / nullptr，看具体题
- **指针 vs 值**：判断节点相同用指针 == 还是 val ==？看题
- **修改原树**：有的题不让改原树，要新建
- **null 检查**：别把 `if (!root)` 漏了

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「二叉树」相关关键词（二叉树）的笔记，按来源分组。

### 灵茶山艾府和代码随想录
- [[二叉树|课程/灵茶山艾府和代码随想录/二叉树.md]]

### 左程云
- [[二叉树的调整|课程/左程云/001. 难点重点照顾/二叉树的调整.md]]
- [[二叉树概念汇总|课程/左程云/002. 知识点总结/20. 算法知识点/二叉树概念汇总.md]]
- [[搜索二叉树|课程/左程云/002. 知识点总结/20. 算法知识点/搜索二叉树.md]]
- [[线索二叉树|课程/左程云/002. 知识点总结/20. 算法知识点/线索二叉树.md]]
- [[Morris遍历解决二叉树问题|课程/左程云/003. 编程基础技巧及工具/Morris遍历解决二叉树问题.md]]
- [[树型dp解决二叉树问题|课程/左程云/003. 编程基础技巧及工具/树型dp解决二叉树问题.md]]

### labuladong
- [[二叉树|课程/labuladong/核心刷题框架/二叉树.md]]

### C++算法
- [[二叉树经典问题|课程/C++算法/必备/二叉树经典问题.md]]

## 参考资料

- [[../../课程/灵茶山艾府和代码随想录/二叉树]] — 你整理的二叉树专题
- [[../../课程/C++算法/必备/二叉树经典问题]] — 你的 C++ 二叉树笔记
- [[../../题目/0437-路径总和-III]] — 路径总和 III 解法
- [[../../课程/labuladong/核心刷题框架]] — labuladong 二叉树思维

## 关联条目

- 上层：[[ds-overview]]
- 衍生：[[bst]] · [[heap]] · [[segment-tree]]
- 算法：[[bfs-dfs]] · [[dp-tree]]
- 思维：[[recursion]]
