---
title: 队列
aliases: [Queue, FIFO, 先进先出, 双端队列, deque]
tags: [data-structure, linear]
created: 2026-05-03
updated: 2026-05-03
---

# 队列

> **TL;DR**：队列 = **先进先出（FIFO）** 的线性结构。push、pop、front 都是 O(1)。是 [[bfs-dfs|BFS]] 的标配，也是 [[monotonic-queue|单调队列]] 的底座。

## 三个核心操作

```cpp
queue<int> q;
q.push(x);             // 入队（队尾）
int f = q.front();     // 看队首（不弹）
q.pop();               // 弹出队首
bool empty = q.empty();
int size = q.size();
```

C++ STL 的 `std::queue` 底层是 `deque`。

## 双端队列 deque

可以两端都 push/pop：

```cpp
deque<int> dq;
dq.push_back(x);   dq.pop_back();
dq.push_front(x);  dq.pop_front();
dq.front();        dq.back();
dq[i];             // 也支持随机访问 O(1)
```

deque 是个**奇怪但有用**的结构：

- 两端 push/pop 都 O(1)
- 支持 `[]` 随机访问 O(1)（但底层是分段数组，常数比 vector 大）
- **是 BFS、单调队列的标配**

## 优先队列（堆）

形式上是队列，但**不按入队顺序**而是**按优先级**出队。详见 [[heap]]。

## 何时想到队列

| 信号 | 应用 |
|---|---|
| "层序遍历" | [[binary-tree\|二叉树]] BFS |
| "最短路径（无权图）" | BFS |
| "状态搜索 / 最少步数" | BFS（迷宫、单词接龙） |
| "滑窗最大 / 最小" | [[monotonic-queue\|单调队列]] |
| "调度任务" | 优先队列 |

## 经典应用

### 1. BFS 层序遍历

```cpp
vector<vector<int>> levelOrder(TreeNode* root) {
    vector<vector<int>> res;
    if (!root) return res;
    queue<TreeNode*> q;
    q.push(root);
    while (!q.empty()) {
        int sz = q.size();           // 重要：固定本层大小
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

### 2. 无权图最短路（BFS）

```cpp
int shortestPath(int s, int t, vector<vector<int>>& g) {
    int n = g.size();
    vector<int> dist(n, -1);
    queue<int> q;
    q.push(s); dist[s] = 0;
    while (!q.empty()) {
        int u = q.front(); q.pop();
        if (u == t) return dist[u];
        for (int v : g[u]) {
            if (dist[v] == -1) {
                dist[v] = dist[u] + 1;
                q.push(v);
            }
        }
    }
    return -1;
}
```

### 3. 单调队列（滑窗最值）

用 deque 维护一个**单调递减队列**，队首始终是窗口最大值：

```cpp
vector<int> maxSlidingWindow(vector<int>& nums, int k) {
    deque<int> dq;             // 存下标，对应值递减
    vector<int> res;
    for (int i = 0; i < nums.size(); ++i) {
        while (!dq.empty() && nums[dq.back()] <= nums[i]) dq.pop_back();
        dq.push_back(i);
        if (dq.front() <= i - k) dq.pop_front();   // 队首过期
        if (i >= k - 1) res.push_back(nums[dq.front()]);
    }
    return res;
}
```

详见 [[monotonic-queue]]。

### 4. 循环队列（手写）

```cpp
class MyCircularQueue {
    vector<int> data;
    int head = 0, tail = 0, sz = 0, cap;
public:
    MyCircularQueue(int k) : data(k), cap(k) {}
    bool enQueue(int x) {
        if (sz == cap) return false;
        data[tail] = x;
        tail = (tail + 1) % cap;
        sz++;
        return true;
    }
    bool deQueue() {
        if (sz == 0) return false;
        head = (head + 1) % cap;
        sz--;
        return true;
    }
    int Front() { return sz ? data[head] : -1; }
    int Rear()  { return sz ? data[(tail - 1 + cap) % cap] : -1; }
};
```

LC 622 经典题。要点：用 `sz` 区分"满"和"空"，避免歧义。

## 复杂度

| 操作 | 复杂度 |
|---|---|
| push / pop / front | O(1) |
| 查找 | O(N) |
| deque 随机访问 | O(1)（常数大于 vector） |

## 队列 vs 栈 vs 优先队列

| 结构 | 出队顺序 | 典型应用 |
|---|---|---|
| 栈 | LIFO | DFS、括号匹配、单调栈 |
| 队列 | FIFO | BFS、层序、滑窗 |
| 优先队列 | 按优先级 | Top-K、Dijkstra、合并 K 链表 |

## 易错点

- C++ `pop()` 不返回值
- 用 `vector` 当队列时 `pop_front` 是 O(N)（要挪所有元素）→ 必须用 `deque` 或 `queue`
- BFS 的"标记已访问"放**入队时**还是**出队时**：通常**入队时**避免重复入队
- 单调队列里存的是**下标**还是**值**？通常是**下标**，方便判断"队首是否过期"

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「队列」相关关键词（队列）的笔记，按来源分组。

### 灵茶山艾府和代码随想录
- [[单调队列|课程/灵茶山艾府和代码随想录/单调队列.md]]
- [[栈和队列|课程/灵茶山艾府和代码随想录/栈和队列.md]]

### 左程云
- [[Java中的容器, 队列, 堆栈|课程/左程云/002. 知识点总结/00. 编程语言相关/Java中的容器, 队列, 堆栈.md]]
- [[优先级队列|课程/左程云/002. 知识点总结/20. 算法知识点/优先级队列.md]]
- [[单调队列|课程/左程云/002. 知识点总结/20. 算法知识点/单调队列.md]]
- [[数组实现栈和队列|课程/左程云/010. 代码模板/数组实现栈和队列.md]]
- [[06 比较器、优先级队列和二叉树|课程/左程云/100. 算法课程/00. 新手班/06 比较器、优先级队列和二叉树.md]]
- [[03 链表结构、栈、队列、递归行为、哈希表和有序表|课程/左程云/100. 算法课程/01. 体系学习班/03 链表结构、栈、队列、递归行为、哈希表和有序表.md]]

### C++算法
- [[堆 优先队列|课程/C++算法/必备/堆 优先队列.md]]

### 题目
- [[0225-用队列实现栈|题目/0225-用队列实现栈.md]]
- [[0232-用栈实现队列|题目/0232-用栈实现队列.md]]
- [[0406-根据身高重建队列|题目/0406-根据身高重建队列.md]]
- [[1944-队列中可以看到的人数|题目/1944-队列中可以看到的人数.md]]

## 参考资料

- [[../../课程/灵茶山艾府和代码随想录/栈和队列]] — 你整理的栈队列专题
- [[../../课程/灵茶山艾府和代码随想录/单调队列]] — 单调队列专题
- [[../../语言/C++/STL]] — `std::queue` / `std::deque`

## 关联条目

- 上层：[[ds-overview]]
- 兄弟：[[stack]] · [[heap]]
- 衍生：[[monotonic-queue]]
- 应用：[[bfs-dfs]] · [[sliding-window]]
