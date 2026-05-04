---
title: 栈
aliases: [Stack, LIFO, 后进先出]
tags: [data-structure, linear]
created: 2026-05-03
updated: 2026-05-03
---

# 栈

> **TL;DR**：栈 = **后进先出（LIFO）** 的线性结构。push、pop、top 都是 O(1)。是 [[recursion|递归]] 的硬件镜像，也是 [[monotonic-stack|单调栈]]、表达式求值、括号匹配的基础。

## 三个核心操作

```cpp
stack<int> st;
st.push(x);              // 入栈
int top = st.top();      // 看栈顶（不弹）
st.pop();                // 弹出栈顶（无返回值！）
bool empty = st.empty();
int size = st.size();
```

注意 C++ 的 `pop()` 不返回值——要先 `top()` 再 `pop()`。

## 实现方式

- **基于数组**：`vector<int>` 当栈用，`push_back` / `pop_back` / `back()`
- **基于链表**：链表头当栈顶（push 头插、pop 头删）

C++ STL 的 `std::stack` 默认底层是 `deque`，可以指定 vector：

```cpp
std::stack<int, std::vector<int>> st;
```

竞赛常用 `vector` 直接当栈，省常数。

## 何时想到栈

信号词 / 模式：

| 信号 | 应用 |
|---|---|
| "嵌套结构" | 括号匹配、HTML 解析 |
| "撤销 / undo" | 编辑器、回溯 |
| "上一个 / 下一个 …" | [[monotonic-stack\|单调栈]] |
| "DFS 改迭代" | 显式开栈 |
| "表达式 / 中缀转后缀" | 调度场算法 |
| "函数调用模拟" | 计算器 |

## 经典应用

### 1. 括号匹配（LC 20）

```cpp
bool isValid(string s) {
    stack<char> st;
    for (char c : s) {
        if (c == '(' || c == '[' || c == '{') st.push(c);
        else {
            if (st.empty()) return false;
            char top = st.top(); st.pop();
            if ((c == ')' && top != '(') ||
                (c == ']' && top != '[') ||
                (c == '}' && top != '{')) return false;
        }
    }
    return st.empty();
}
```

### 2. 中缀表达式求值（带优先级）

调度场算法（Shunting Yard）：用一个数字栈 + 一个运算符栈，遇到优先级更低的运算符就先消化栈顶高优先级的。

### 3. DFS 改迭代

```cpp
void dfs(int start, vector<vector<int>>& g) {
    stack<int> st;
    st.push(start);
    vector<bool> vis(g.size(), false);
    while (!st.empty()) {
        int u = st.top(); st.pop();
        if (vis[u]) continue;
        vis[u] = true;
        for (int v : g[u]) if (!vis[v]) st.push(v);
    }
}
```

注意和递归 DFS 的区别：迭代版可能先访问后入栈的邻居。

### 4. 单调栈

栈中维护一个**单调**序列（递增或递减），用于求"下一个更大 / 更小元素"问题。

详见 [[monotonic-stack]]。代表题：[[problems/trapping-rain-water]] · [[problems/largest-rectangle]]。

### 5. 最小栈（LC 155）

要求 push / pop / top / getMin 都 O(1)。两种做法：

- **辅助栈**：每次 push 时，min 栈也 push 当前 min
- **差值栈**：栈中存 `val - min`

## 复杂度

| 操作 | 复杂度 |
|---|---|
| push / pop / top | O(1) |
| 查找特定值 | O(N) |

## 与递归的关系

每次函数调用都在**调用栈**上压一帧：保存返回地址、局部变量。**任何递归算法都可以改写成显式栈的迭代算法**。这是处理"递归太深爆栈"的标准手段。

```cpp
// 递归 DFS 中序
void inorder(TreeNode* root) {
    if (!root) return;
    inorder(root->left);
    visit(root);
    inorder(root->right);
}

// 迭代 DFS 中序
void inorder(TreeNode* root) {
    stack<TreeNode*> st;
    auto* cur = root;
    while (cur || !st.empty()) {
        while (cur) { st.push(cur); cur = cur->left; }
        cur = st.top(); st.pop();
        visit(cur);
        cur = cur->right;
    }
}
```

## 易错点

- C++ `pop()` 不返回值
- 空栈调 `top()` 是 UB（未定义行为）→ 一定先 `empty()`
- 栈和队列方向反了
- 用栈做 BFS 是错的；用队列做 DFS 也是错的

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「栈」相关关键词（栈）的笔记，按来源分组。

### 灵茶山艾府 + 代码随想录
- [[课程/灵茶山艾府和代码随想录/单调栈|课程/灵茶山艾府 + 代码随想录/单调栈.md]]
- [[栈和队列|课程/灵茶山艾府 + 代码随想录/栈和队列.md]]

### 左程云
- [[_单调栈在相等情况下做错化处理|课程/左程云/001. 难点重点照顾/_单调栈在相等情况下做错化处理.md]]
- [[Java中的容器, 队列, 堆栈|课程/左程云/002. 知识点总结/00. 编程语言相关/Java中的容器, 队列, 堆栈.md]]
- [[单调栈是什么|课程/左程云/002. 知识点总结/20. 算法知识点/单调栈是什么.md]]
- [[滑动窗口 & 单调栈|课程/左程云/002. 知识点总结/20. 算法知识点/滑动窗口 & 单调栈.md]]
- [[课程/灵茶山艾府和代码随想录/单调栈|课程/左程云/003. 编程基础技巧及工具/单调栈.md]]
- [[单调栈代码|课程/左程云/010. 代码模板/单调栈代码.md]]

## 参考资料

- [[../../课程/灵茶山艾府 + 代码随想录/栈和队列]] — 你整理的栈队列专题
- [[../../语言/C++/STL]] — `std::stack` / `std::deque`
- [[../../课程/C++算法/必备/堆 优先队列]] — 相关数据结构

## 关联条目

- 上层：[[ds-overview]]
- 兄弟：[[queue]]
- 衍生：[[monotonic-stack]]
- 关联：[[recursion]] · [[bfs-dfs]]
- 应用：[[problems/trapping-rain-water]] · [[problems/largest-rectangle]]
