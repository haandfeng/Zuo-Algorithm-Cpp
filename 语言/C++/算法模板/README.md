---
title: C++ 算法模板库
updated: 2026-05-04
---

# C++ 算法模板库

> **可直接复制粘贴**的算法模板代码。和 [`algorithm-wiki/wiki/`](../../../algorithm-wiki/wiki/) 是互补关系：
> - **wiki/** 解释「为什么这样做」（直觉 + 何时想到 + 复杂度 + 易错点）
> - **本目录** 提供「直接能编译跑的代码」

## 与现有模板的关系

| 来源 | 内容 | 何时用 |
|---|---|---|
| [`课程/左程云/010. 代码模板/`](../../../课程/左程云/010.%20代码模板/) | 左程云课程**自带**的模板 | 学课程时直接用 |
| [`课程/C++算法/必备/`](../../../课程/C++算法/必备/) | 你早期整理的专题（含部分模板） | 历史参考 |
| **本目录（语言/C++/算法模板/）** | 你**自己沉淀** + **优化过**的模板 | 刷题时即取即用 |
| [`algorithm-wiki/wiki/`](../../../algorithm-wiki/wiki/) | 概念解释 + 模板代码（教学用） | 复习/讲解时 |

## 推荐组织

每个模板一个 `.cpp` 或 `.md` 文件：

```
算法模板/
├── README.md                     ← 你正在读的文件
│
├── 数据结构/
│   ├── 并查集.cpp
│   ├── 树状数组.cpp
│   ├── 线段树-区间求和.cpp
│   ├── 线段树-区间最值.cpp
│   ├── 字典树.cpp
│   └── ...
│
├── 算法/
│   ├── 快排.cpp
│   ├── 归并排序.cpp
│   ├── KMP.cpp
│   ├── Manacher.cpp
│   ├── Dijkstra.cpp
│   ├── Bellman-Ford.cpp
│   └── ...
│
├── 技巧/
│   ├── 单调栈.cpp
│   ├── 单调队列.cpp
│   ├── 滑动窗口-不定长.cpp
│   ├── 二分答案.cpp
│   └── ...
│
└── 范式/
    ├── 回溯-子集.cpp
    ├── 回溯-排列.cpp
    ├── 回溯-组合.cpp
    ├── 0-1背包.cpp
    └── ...
```

## 文件格式约定

每个模板文件按这个骨架：

```cpp
/**
 * 模板：单调栈（求每个元素右侧第一个更大）
 * 复杂度：时间 O(N)，空间 O(N)
 * 适用：见 [[../../../algorithm-wiki/wiki/techniques/monotonic-stack]]
 *
 * 关键参数：
 * - nums: 输入数组
 * 返回：每个位置的"右侧第一个更大"下标，没有则 -1
 *
 * 测试题：LC 496 / 503 / 739 / 901
 */
#include <vector>
#include <stack>
using namespace std;

vector<int> nextGreater(vector<int>& nums) {
    int n = nums.size();
    vector<int> ans(n, -1);
    stack<int> st;
    for (int i = 0; i < n; ++i) {
        while (!st.empty() && nums[i] > nums[st.top()]) {
            ans[st.top()] = i;
            st.pop();
        }
        st.push(i);
    }
    return ans;
}
```

## 怎么开始填

**不必现在就写所有模板**——刷到对应的题再回来沉淀：

1. 刷题时，如果某段代码"我以后还想再用"
2. 提取成模板，加上注释 + 复杂度 + 测试题号
3. 放到对应子目录
4. 在 `algorithm-wiki/wiki/` 对应概念文章里反向引用

## 关联

- [`algorithm-wiki/wiki/index.md`](../../../algorithm-wiki/wiki/index.md) — 概念入口
- [`课程/左程云/010. 代码模板/`](../../../课程/左程云/010.%20代码模板/) — 现成模板
- [`USAGE.md`](../../../USAGE.md) — 仓库使用指南
