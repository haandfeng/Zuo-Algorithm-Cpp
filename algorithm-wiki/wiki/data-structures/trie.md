---
title: Trie
aliases: [字典树, 前缀树, Prefix Tree]
tags: [data-structure, tree, string]
created: 2026-05-03
updated: 2026-05-03
---

# Trie（字典树）

> **TL;DR**：Trie 是把**一组字符串**按**字符作为边**组织成的树。每个节点是一个字符位置，从根到某节点的路径就是一个字符串前缀。**前缀查询、计数、最大异或对**的标配。

## 直观例子

把 `["app", "apple", "apply", "bat"]` 存进 Trie：

```text
                  (root)
                 /      \
                a        b
                |        |
                p        a
                |        |
                p ✓      t ✓
                |
                l
               / \
              e ✓ y ✓
```

- 节点上的 `✓` 表示"到这里是一个完整单词"
- 共享前缀的字符串共享路径 → 节省空间

## 实现（数组版，固定字符集）

26 字母小写：

```cpp
struct Trie {
    struct Node {
        int next[26] = {0};       // 下一字符的子节点下标
        bool end = false;
        int cnt = 0;              // 经过该节点的字符串数（前缀计数）
    };
    vector<Node> nodes;
    Trie() : nodes(1) {}          // root = nodes[0]

    void insert(const string& s) {
        int u = 0;
        for (char c : s) {
            int i = c - 'a';
            if (nodes[u].next[i] == 0) {
                nodes.emplace_back();
                nodes[u].next[i] = nodes.size() - 1;
            }
            u = nodes[u].next[i];
            nodes[u].cnt++;
        }
        nodes[u].end = true;
    }

    bool search(const string& s) {
        int u = 0;
        for (char c : s) {
            int i = c - 'a';
            if (nodes[u].next[i] == 0) return false;
            u = nodes[u].next[i];
        }
        return nodes[u].end;
    }

    bool startsWith(const string& prefix) {
        int u = 0;
        for (char c : prefix) {
            int i = c - 'a';
            if (nodes[u].next[i] == 0) return false;
            u = nodes[u].next[i];
        }
        return true;
    }
};
```

## 实现（指针版）

更"教科书"风格，但容易内存碎片：

```cpp
struct TrieNode {
    bool end = false;
    TrieNode* child[26] = {nullptr};
};

class Trie {
    TrieNode* root = new TrieNode();
public:
    void insert(string s) {
        TrieNode* cur = root;
        for (char c : s) {
            int i = c - 'a';
            if (!cur->child[i]) cur->child[i] = new TrieNode();
            cur = cur->child[i];
        }
        cur->end = true;
    }
    // search / startsWith 同理
};
```

竞赛通常用数组版，工程可以用指针版。

## 复杂度

| 操作 | 复杂度 |
|---|---|
| insert / search / startsWith | O(L)，L 是字符串长度 |
| 空间 | O(N · L · Σ)，Σ 是字符集大小 |

**空间敏感**：26 字母小写已经占 26 个 int 每节点；扩展到 unicode 必须用哈希子节点 `unordered_map<char, Node*>`。

## 何时想到 Trie

| 信号 | 应用 |
|---|---|
| "前缀查询 / 计数" | 自动补全、敏感词检测 |
| "最大 / 最小异或对" | 0-1 Trie |
| "拼写纠错" | Trie + 编辑距离 |
| "多模式串匹配" | AC 自动机（Trie + KMP） |
| "字典 / 词典" | Trie 比 hash + 排序更省空间 |

## 经典题型

| 题 | 用法 |
|---|---|
| 实现 Trie LC 208 | 模板 |
| 添加与搜索单词 LC 211 | Trie + 通配符 `.` 用 DFS |
| 单词搜索 II LC 212 | 把字典建 Trie，再 DFS 网格剪枝 |
| 数组中两个数的最大异或值 LC 421 | 0-1 Trie |
| 实现一个磁盘自动补全 | Trie + DFS 输出后缀 |
| 数组中第 K 大异或对（CF / 牛客） | 0-1 Trie |
| 最长公共前缀 LC 14 | 不一定要 Trie，但 Trie 解直观 |

### 0-1 Trie 模板（最大异或对）

```cpp
struct BitTrie {
    int next[3000005][2] = {0};
    int cnt = 0;

    void insert(int x) {
        int u = 0;
        for (int i = 30; i >= 0; --i) {
            int b = (x >> i) & 1;
            if (!next[u][b]) next[u][b] = ++cnt;
            u = next[u][b];
        }
    }

    int queryMaxXor(int x) {
        int u = 0, res = 0;
        for (int i = 30; i >= 0; --i) {
            int b = (x >> i) & 1;
            if (next[u][b ^ 1]) { res |= (1 << i); u = next[u][b ^ 1]; }
            else u = next[u][b];
        }
        return res;
    }
};
```

## Trie vs HashMap

| 维度 | Trie | HashMap |
|---|---|---|
| 前缀查询 | ✅ O(L) | ❌（要遍历） |
| 全字串查询 | O(L) | O(L) |
| 内存 | 多 | 少 |
| 字符集大 | 退化 | 不影响 |

## 易错点

- 节点数估算：N 个串 × L 平均长度 = 节点上界
- 数组版数组大小开够（不够会越界）
- `end` 和"前缀存在"是两件事：`startsWith` 是前缀，`search` 才是完整词
- 字符集不一定是小写字母 → 改 `next[]` 大小或用 hash

## 参考资料

- [[../../算法笔记/zuoAlgorithm/100. 算法课程]] — 左程云 Trie 章节
- [[../../labuladong/经典数据结构算法]] — labuladong Trie

## 关联条目

- 上层：[[ds-overview]]
- 兄弟：[[hash-table]]
- 应用：[[kmp]]（多模式版即 AC 自动机）
