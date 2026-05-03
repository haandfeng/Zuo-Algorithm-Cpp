---
title: 回溯
aliases: [Backtracking, 回溯算法, DFS 暴搜]
tags: [pattern, dfs]
created: 2026-05-03
updated: 2026-05-03
---

# 回溯

> **TL;DR**：回溯 = "试错 + 撤销"。本质是 [[bfs-dfs|DFS]] 在**多叉决策树**上的应用。三大类：**子集 / 组合 / 排列**，外加棋盘类（N 皇后、数独）。

## 通用模板

```cpp
void backtrack(状态 path, 选择列表 choices) {
    if (满足终止条件) {
        收集答案;
        return;
    }
    for (选择 c : choices) {
        if (!合法(c)) continue;
        做选择(path, c);
        backtrack(path, 新选择列表);
        撤销选择(path, c);             // ← 回溯的"回"
    }
}
```

**核心**：在递归调用前**做选择**、调用返回后**撤销选择**——保证每条 DFS 路径独立。

## 三大题型

### 1. 子集（Subsets）

每个元素**选**或**不选**：

```cpp
vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>> ans;
    vector<int> path;
    function<void(int)> dfs = [&](int i) {
        if (i == (int)nums.size()) { ans.push_back(path); return; }
        // 不选 nums[i]
        dfs(i + 1);
        // 选 nums[i]
        path.push_back(nums[i]);
        dfs(i + 1);
        path.pop_back();
    };
    dfs(0);
    return ans;
}
```

或用"枚举起点"框架：

```cpp
function<void(int)> dfs = [&](int start) {
    ans.push_back(path);
    for (int i = start; i < (int)nums.size(); ++i) {
        path.push_back(nums[i]);
        dfs(i + 1);
        path.pop_back();
    }
};
dfs(0);
```

### 2. 组合（Combinations）

从 N 个里选 K 个，**有序选**避免重复：

```cpp
vector<vector<int>> combine(int n, int k) {
    vector<vector<int>> ans;
    vector<int> path;
    function<void(int)> dfs = [&](int start) {
        if ((int)path.size() == k) { ans.push_back(path); return; }
        for (int i = start; i <= n; ++i) {
            // 剪枝：剩余元素不够凑齐 k 个
            if (n - i + 1 < k - (int)path.size()) break;
            path.push_back(i);
            dfs(i + 1);
            path.pop_back();
        }
    };
    dfs(1);
    return ans;
}
```

### 3. 排列（Permutations）

每个元素都要用、顺序不同算不同：

```cpp
vector<vector<int>> permute(vector<int>& nums) {
    int n = nums.size();
    vector<vector<int>> ans;
    vector<int> path;
    vector<bool> used(n, false);
    function<void()> dfs = [&]() {
        if ((int)path.size() == n) { ans.push_back(path); return; }
        for (int i = 0; i < n; ++i) {
            if (used[i]) continue;
            used[i] = true;
            path.push_back(nums[i]);
            dfs();
            path.pop_back();
            used[i] = false;
        }
    };
    dfs();
    return ans;
}
```

## 去重技巧（含重复元素）

输入有重复时，要先排序再加"同层去重"：

```cpp
sort(nums.begin(), nums.end());
for (int i = start; i < n; ++i) {
    if (i > start && nums[i] == nums[i - 1]) continue;     // 同层跳过相同
    // ...
}
```

排列去重：`if (i > 0 && nums[i] == nums[i - 1] && !used[i - 1]) continue;`

LC 47 全排列 II、LC 90 子集 II 都是这个套路。

## 棋盘 / 网格类

### N 皇后（LC 51）

```cpp
vector<vector<string>> solveNQueens(int n) {
    vector<vector<string>> ans;
    vector<string> board(n, string(n, '.'));
    vector<bool> col(n), diag1(2*n), diag2(2*n);
    function<void(int)> dfs = [&](int row) {
        if (row == n) { ans.push_back(board); return; }
        for (int c = 0; c < n; ++c) {
            if (col[c] || diag1[row + c] || diag2[row - c + n]) continue;
            board[row][c] = 'Q';
            col[c] = diag1[row + c] = diag2[row - c + n] = true;
            dfs(row + 1);
            col[c] = diag1[row + c] = diag2[row - c + n] = false;
            board[row][c] = '.';
        }
    };
    dfs(0);
    return ans;
}
```

### 单词搜索（LC 79）

网格 DFS + visited，找完一个方向回溯到另一个：

```cpp
bool exist(vector<vector<char>>& g, string word) {
    int n = g.size(), m = g[0].size();
    function<bool(int,int,int)> dfs = [&](int x, int y, int k) -> bool {
        if (k == (int)word.size()) return true;
        if (x < 0 || x >= n || y < 0 || y >= m) return false;
        if (g[x][y] != word[k]) return false;
        char c = g[x][y]; g[x][y] = '#';                  // 临时标记
        bool ok = dfs(x+1,y,k+1) || dfs(x-1,y,k+1) || dfs(x,y+1,k+1) || dfs(x,y-1,k+1);
        g[x][y] = c;                                       // 撤销
        return ok;
    };
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < m; ++j)
            if (dfs(i, j, 0)) return true;
    return false;
}
```

## 剪枝（Pruning）

**好的剪枝能让 O(2^N) 在小规模实际可跑**：

| 类型 | 例子 |
|---|---|
| 可行性剪枝 | 当前已不可能满足条件，直接 return |
| 最优性剪枝 | 当前路径已比已知最优差，return |
| 重复性剪枝 | 同层去重（见上） |
| 顺序性剪枝 | 强制有序，避免对称 |
| 边界剪枝 | 越界 / 触墙立即 return |
| 记忆化 | 子问题重复 → 转 [[dp\|DP]] |

## 何时想到回溯

| 信号 | 例子 |
|---|---|
| "返回所有 …" | 所有路径、所有子集 |
| "全部排列 / 组合 / 子集" | 三大模板 |
| "N 皇后 / 数独" | 棋盘 |
| "划分 K 个相等子集" LC 698 | 回溯 |
| "解括号 / 解 IP / 解电话号码" | 暴搜 |
| "状态空间 < 几百万" | 回溯（更大就要 [[dp\|DP]]） |

如果是**"求方案数 / 求最值"** 而不是**"列出所有"**，通常应该转 [[dp]]。

## 经典题型

| 题 | 类型 |
|---|---|
| 全排列 LC 46 | 排列 |
| 全排列 II LC 47 | 排列 + 去重 |
| 子集 LC 78 | 子集 |
| 子集 II LC 90 | 子集 + 去重 |
| 组合 LC 77 | 组合 |
| 组合总和 LC 39 | 组合 + 可重复 |
| 组合总和 II LC 40 | 组合 + 去重 |
| 电话号码的字母组合 LC 17 | 多叉 |
| 括号生成 LC 22 | 剪枝 |
| 复原 IP 地址 LC 93 | 划分 |
| 分割回文串 LC 131 | 划分 |
| 单词搜索 LC 79 | 网格 |
| N 皇后 LC 51 | 棋盘 |
| 数独 LC 37 | 棋盘 |
| 划分为 k 个相等子集 LC 698 | 状压 + 回溯 |

## 易错点

- 忘记**撤销选择** → 状态污染
- 收集答案时**没拷贝**——`ans.push_back(path)` 是值拷贝没问题，但用引用就会被修改
- "选 / 不选" 顺序：先 dfs 再 push 等价于先放 root 再 dfs（前序）
- 同层去重 `i > start` 还是 `i > 0`？画图区分**同层**和**同枝**
- 死循环：递归没正确推进 i / start

## 参考资料

- [[../../课程/灵茶山艾府 + 代码随想录/回溯]] — 你的回溯专题
- [[../../课程/labuladong/经典暴力搜索算法]] — labuladong 回溯框架

## 关联条目

- 上层：[[algorithm-overview]]
- 思维：[[recursion]] · [[bfs-dfs]]
- 数据结构：[[stack]]
- 兄弟：[[dp]]（"列举" → 回溯；"计数 / 最值" → DP）
