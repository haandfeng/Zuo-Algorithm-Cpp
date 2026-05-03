---
title: 递归
aliases: [Recursion, 递归思维, 递归三问]
tags: [concept, foundation]
created: 2026-05-03
updated: 2026-05-03
---

# 递归

> **TL;DR**：递归 = 把"规模为 N 的问题"用"规模更小的同类问题"表达，再加上 base case 终止。**写递归不是写循环，是定义函数的语义**。

## 递归 = 数学归纳

写递归时**别想"它怎么展开"**，那是死胡同。要想：

1. **它是干什么的**（函数语义）
2. **base case**（最小问题怎么直接答）
3. **递推**（假设小问题已经解决，怎么用它推大问题）

这就是**数学归纳法**：基础情况 + 假设 P(k) → 推 P(k+1)。

## 模板：左程云"递归三问"

每写一个递归函数，明确这三件事：

```cpp
// 问题：返回某棵二叉树的最大深度
int maxDepth(TreeNode* root) {
    // ① 函数语义：返回以 root 为根的子树的最大深度
    // ② base case
    if (!root) return 0;
    // ③ 递推：假设左/右子树深度已知，整棵树深度 = max + 1
    return max(maxDepth(root->left), maxDepth(root->right)) + 1;
}
```

写之前先在注释里明确这三行，能省掉一半 debug 时间。

## 五种常见递归形态

### 1. 线性递归（尾递归友好）

```cpp
int sum(int n) {
    if (n == 0) return 0;
    return n + sum(n - 1);   // 每次只调一次
}
```

复杂度 O(N)、栈深 O(N)。可改为循环。

### 2. 二叉递归

```cpp
int fib(int n) {
    if (n < 2) return n;
    return fib(n - 1) + fib(n - 2);   // 调两次
}
```

如果不剪枝是 O(2^N)，加 [[dp|备忘录]] 后 O(N)。

### 3. 树形递归（最常见）

```cpp
int maxDepth(TreeNode* root);   // 上面那个例子
```

每个节点调一次，总 O(N)。

### 4. 多路递归 / 回溯

```cpp
void backtrack(vector<int>& path, int start) {
    if (满足结束条件) { 收集答案; return; }
    for (int i = start; i < n; ++i) {
        path.push_back(i);
        backtrack(path, i + 1);
        path.pop_back();           // 撤销选择
    }
}
```

详见 [[backtracking]]。

### 5. 互递归

```cpp
bool isEven(int n) { return n == 0 ? true : isOdd(n - 1); }
bool isOdd(int n)  { return n == 0 ? false : isEven(n - 1); }
```

实战很少见，但要知道这种形态。

## 递归 vs 迭代

| 维度 | 递归 | 迭代 |
|---|---|---|
| 思维 | 自顶向下 | 自底向上 |
| 代码 | 简洁、贴近问题定义 | 啰嗦、显式管理状态 |
| 性能 | 常数大（栈帧） | 常数小 |
| 栈空间 | O(深度) | O(1) |
| 可读 | 树/图问题更清晰 | 数组/字符串更清晰 |

**经验**：
- 树 / 图问题首选递归；难写就改 BFS
- 数组 / 字符串问题首选迭代
- DP 问题：记忆化递归（top-down）vs 迭代填表（bottom-up）按熟悉程度选

## 改成迭代

任何递归都可以**显式开一个 stack** 改成迭代：

```cpp
// 递归版
void preorder(TreeNode* root) {
    if (!root) return;
    visit(root);
    preorder(root->left);
    preorder(root->right);
}

// 迭代版
void preorder(TreeNode* root) {
    if (!root) return;
    stack<TreeNode*> st;
    st.push(root);
    while (!st.empty()) {
        auto* node = st.top(); st.pop();
        visit(node);
        if (node->right) st.push(node->right);   // 右先入，左后入（栈是 LIFO）
        if (node->left)  st.push(node->left);
    }
}
```

**为什么有时要这么改**：递归深度过深会爆栈（C++ 默认栈 1MB，深度 ~1e5 就危险）。

## 写递归的常见 BUG

1. **忘 base case**：无限递归，stack overflow
2. **base 写错**：例如 `n == 1` 写成 `n == 0`，差 1 错位
3. **改了不该改的状态**：回溯时忘恢复 → 状态污染
4. **多次调用同子问题**：例如朴素 fib，没加 memo
5. **return 漏写**：递归返回值没 return 出去
6. **左右子问题混淆**：`return solve(left) + solve(right)` 写成 `solve(left); return solve(right)`

## 递归的算力代价

每次调用要做：

- 压栈：保存返回地址、寄存器
- 传参：拷贝实参（C++ 引用 / 指针可避免大对象拷贝）
- 弹栈：恢复

所以**递归的常数 ≈ 普通函数调用的 1.5~3 倍**。在算法竞赛极限场景下值得改迭代；面试通常不需要。

## 与其它概念的关系

- 是 [[divide-and-conquer]] 的实现工具
- 是 [[dp]] 的"top-down 写法"载体（记忆化）
- 是 [[backtracking]] 的核心
- 是 [[bfs-dfs|DFS]] 的天然写法
- 树 / 图问题中无处不在

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「递归」相关关键词（递归）的笔记，按来源分组。

### 左程云
- [[_暴力递归改动态规划的套路|课程/左程云/001. 难点重点照顾/_暴力递归改动态规划的套路.md]]
- [[二叉树的递归套路|课程/左程云/004. 刷题经验总结/二叉树的递归套路.md]]
- [[二叉树的递归套路_空树的处理|课程/左程云/004. 刷题经验总结/二叉树的递归套路_空树的处理.md]]
- [[暴力递归及动态规划中的模型尝试过程|课程/左程云/004. 刷题经验总结/暴力递归及动态规划中的模型尝试过程.md]]
- [[设计暴力递归过程的原则跟模型|课程/左程云/004. 刷题经验总结/设计暴力递归过程的原则跟模型.md]]
- [[递归能力练习题目|课程/左程云/004. 刷题经验总结/递归能力练习题目.md]]

### C++算法
- [[嵌套类问题的递归解体套路|课程/C++算法/必备/嵌套类问题的递归解体套路.md]]
- [[常见递归题目|课程/C++算法/必备/常见递归题目.md]]

## 参考资料

- [[../../课程/labuladong/核心刷题框架]] — labuladong 的"递归思维"
- [[../../课程/左程云/001. 难点重点照顾]] — 左程云递归章节

## 关联条目

- 上层：[[algorithm-overview]]
- 配套：[[divide-and-conquer]] · [[backtracking]] · [[bfs-dfs]] · [[dp]]
- 数据结构：[[binary-tree]]
