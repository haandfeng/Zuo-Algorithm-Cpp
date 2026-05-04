---
audience: LLM agent
purpose: operating manual for ingest / compile / lint / Q&A on the algorithm wiki
---

# 工作流与协议（LLM 操作手册）

本文档面向**操作这个算法 Wiki 的 LLM**。人类用户也可以读，了解系统在做什么。

---

## 1. Ingest：把素材编译进 wiki/

### 触发条件

- 用户："ingest 这道题 / 这篇专题 / 这个章节"、"把刚刷的 xxx.md 整理一下"
- 或脚本：`python scripts/ingest.py <path>`

### 步骤

1. **登记原文**：在 [`MEMORY.md`](MEMORY.md) 的 `## 原始素材索引` 追加 `- [标题](相对路径) — 一句话摘要`
2. **抽取候选概念**：识别 2~10 个**已存在或应当存在**于 `wiki/` 的概念，给出对照（已有 / 新建）
3. **更新已有条目**：在 `wiki/xxx.md` 合适位置加新事实，并在文末 `参考资料` 加来源
4. **创建新条目**：在合适子目录新建 `xxx.md`，遵循 [文章模板](#文章模板)
5. **建立反向链接**：每个新条目至少与 2 个已有条目互相 `[[link]]`，避免孤立节点
6. **更新 MOC**：如果是一级主题，在对应的 `wiki/maps/*.md` 中加入
7. **报告**：向用户输出简短 ingest 报告（更新的条目 / 新建的条目 / 待人类确认的歧义）

### 反例

- ❌ 把整篇原文复制到 `wiki/`
- ❌ 在多处重复同一个概念定义
- ❌ 创建只有标题的占位 stub

---

## 2. Q&A：基于 wiki 回答

### 协议

1. **先看索引**：从 [`wiki/index.md`](wiki/index.md) 和 `wiki/maps/` 出发
2. **必要时检索**：调用 `python scripts/search.py "关键词"` 或 `grep -ri`
3. **必要时回 raw/**：信息密度不够时回查原始题目 / 课程笔记
4. **回答给引用**：每个关键论点末尾用 `（见 [[monotonic-stack#经典模板]]）` 标注
5. **发现缺失 / 矛盾**：在回答末尾另起 "📌 Wiki 维护建议" 列出来

### 输出形式

- **简短文本**：默认
- **`outputs/reports/yyyy-mm-dd-题目.md`**：长答 / 综述 / 对比表
- **`outputs/slides/yyyy-mm-dd-题目.md`**：Marp 幻灯片
- **mermaid / matplotlib**：图示（保存到 `raw/images/` 并在文章中引用）

---

## 3. Lint：定期体检

`python scripts/lint.py` 触发，或人类手动要求。

需要检测：

| 检查项 | 含义 |
|---|---|
| 断链 | `[[link]]` 指向不存在的文件 |
| 孤立节点 | 没有任何反向链接的条目（除 index/MOC 外） |
| 重复定义 | 同一概念在多处展开 |
| 信息矛盾 | 不同条目对同一事实有冲突描述 |
| 缺失条目 | 多次被 `[[xxx]]` 引用但 `xxx.md` 不存在 |
| 模板缺漏 | 没有 frontmatter / 没有 `参考资料` 段 |
| 复杂度错误 | "O(N) 时间 O(N) 空间" 这类断言要复核 |
| 代码片段编译性 | C++ 模板代码语法正确（可选项） |

输出 `outputs/reports/lint-yyyy-mm-dd.md`，征求人类确认后再修。

---

## 4. Output：派生产物

常见形式：

- **概念地图**（`wiki/maps/*.md`）：mermaid 画某领域全景
- **对比表**：例如 `outputs/reports/单调栈-vs-单调队列.md`
- **题型刷题路线**：例如 `outputs/reports/dp-roadmap.md`
- **Marp 幻灯片**：用于复盘 / 分享
- **题解综述**：例如 "Hot100 中所有用到回溯的题 + 公共骨架"

派生产物如果有持久价值，回流到 `wiki/`；一次性回答放 `outputs/` 即可。

---

## 文章模板

新建 `wiki/<目录>/<slug>.md` 时使用以下骨架：

```markdown
---
title: 单调栈
aliases: [Monotonic Stack, 递增栈, 递减栈]
tags: [technique, stack]
difficulty: medium
created: 2026-05-03
updated: 2026-05-03
---

# 单调栈

> **TL;DR**：一句话定义 + 主用场景。

## 直觉

…

## 何时想到它

- 信号 1：题目要求"下一个更大 / 更小元素"
- 信号 2：…

## 经典模板（C++）

```cpp
// 求每个元素的"右侧第一个比它大"的下标
vector<int> nextGreater(vector<int>& nums) {
    int n = nums.size();
    vector<int> ans(n, -1);
    stack<int> st;                 // 栈中放下标，对应值递减
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

## 复杂度

- 时间 O(N)：每个元素入栈出栈各一次
- 空间 O(N)

## 关联条目

- 上层：[[stack]]
- 兄弟：[[monotonic-queue]]
- 应用：[[problems/trapping-rain-water]] · [[problems/largest-rectangle]]

## 常见变体 / 易错点

…

## 参考资料

- [[课程/灵茶山艾府和代码随想录/单调栈]]
- 左程云：第 N 节，单调栈
- LeetCode 84 / 42 / 496 / 503 / 739
```

每篇文章应当：
- 有 frontmatter（title / aliases / tags / 可选 difficulty / created / updated）
- 一开始有 1~2 句 TL;DR
- 至少 2 个反向 `[[link]]`
- 末尾的 `参考资料` 指向 `raw/` 或 Vault 里具体笔记 / 题目
- 长度控制在 600~3000 字之间，超过就拆分

---

## 风格约定

- **语言**：正文简体中文；类名、变量名、算法英文名、LeetCode 题号保持原样
- **代码**：默认 **C++**；必要时附 Python 对照；标好语言（` ```cpp ` / ` ```python `）
- **复杂度**：写成 `时间 O(N log N)，空间 O(N)`，N 等符号在文中先解释
- **缩写首次出现**：写完整形式 + 缩写，例如「最长递增子序列（LIS, Longest Increasing Subsequence）」
- **链接**：内部用 `[[wikilink]]`；外部 URL 用 `[文字](url)`；指向 Vault 已有笔记用 `[[../子目录/笔记名]]`
- **数学**：能用文字解释优先文字；公式 `$...$` / `$$...$$`
