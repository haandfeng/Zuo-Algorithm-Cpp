---
title: 仓库使用指南
audience: 自己
updated: 2026-05-04
---

# 算法仓库使用指南

> 这份文档是给"未来的自己"看的。打开 Vault 后从这里开始就够用了。

## 目录

- [仓库结构（30 秒看完）](#仓库结构30-秒看完)
- [日常工作流：5 个场景](#日常工作流5-个场景)
  - [场景 1：刷一道新题](#场景-1刷一道新题)
  - [场景 2：复习某个概念](#场景-2复习某个概念)
  - [场景 3：让 LLM 跨主题问答](#场景-3让-llm-跨主题问答)
  - [场景 4：从外部抓新素材](#场景-4从外部抓新素材)
  - [场景 5：每周维护](#场景-5每周维护)
- [Prompt 速查（可直接复制）](#prompt-速查可直接复制)
- [脚本速查](#脚本速查)
- [七个重要约定](#七个重要约定)
- [推荐节奏](#推荐节奏)
- [进阶玩法](#进阶玩法)
- [FAQ](#faq)

---

## 仓库结构（30 秒看完）

```text
左程云算法/                    ← Obsidian Vault 根
│
├── algorithm-wiki/           🧠 概念地图（55 篇短文，LLM 维护，互相 [[link]]）
│   ├── wiki/                    入口：wiki/index.md
│   ├── scripts/                 4 个 Python 脚本：ingest / lint / search / cross_link / compile
│   ├── outputs/                 LLM 生成的报告 / 幻灯片
│   ├── raw/                     外部抓取的原始素材
│   ├── README.md                总入口
│   ├── WORKFLOW.md              LLM 操作手册（让 LLM 读这个）
│   └── MEMORY.md                自动维护的索引
│
├── 课程/                     🎓 按作者分（只读快照）
│   ├── 左程云/                  你最主要的学习来源（学习大纲、模板、960+ 题）
│   ├── labuladong/              核心刷题框架
│   ├── C++算法/                 入门 + 必备
│   └── 灵茶山艾府 + 代码随想录/  按题型专题（17 篇）
│
├── 题单/                     📋 面试题单（先不拆，保留单文件）
│   ├── Hot100.md / Blind75.md / Grind75.md / Leetcode150.md
│   └── 公司Tag/
│
├── 题目/                     📝 你的单题深度笔记（每题一个文件）
│
├── 语言/                     🔧 C++ / Java / Python 参考
│
├── README.md                 仓库说明（已更新）
└── USAGE.md                  ← 你正在读的文件
```

**记住三件事**：

1. **`algorithm-wiki/wiki/`** 是入口，所有概念都在这里
2. **`课程/`** 是只读，不要手动编辑
3. **`题目/`** 是你的主战场，新题都丢这里

---

## 日常工作流：5 个场景

### 场景 1：刷一道新题

**步骤**：

1. 在 LeetCode / 牛客上做完
2. 在 `题目/` 下新建 `0XXX-题名.md`（4 位题号 + 中划线 + 题名）
3. 顶部加 frontmatter：

   ```yaml
   ---
   leetcode: 206
   difficulty: easy
   tags: [linked-list, recursion]
   covers: [linked-list, recursion]   # 这道题用到的概念，未来 LLM 用得上
   date: 2026-05-04
   solved: true
   ---
   ```

4. 写下解题思路 + C++ 代码 + 复杂度分析
5. 登记进 wiki 索引：

   ```bash
   python algorithm-wiki/scripts/ingest.py 题目/0206-反转链表.md
   ```

6. （可选）让 LLM 把它链进相关 wiki 文章——用下面的 [Prompt 1](#1-把新题链入-wiki)

---

### 场景 2：复习某个概念

**步骤**：

1. Obsidian 打开 `algorithm-wiki/wiki/index.md`
2. 找到概念条目，例如 `[[monotonic-stack]]`
3. 读：
   - **TL;DR**（1 句话）
   - **直觉 / 何时想到 / 经典模板（C++）**
   - **「📂 我 Vault 里的相关笔记」段** ← 看这里！会按"灵神 / Carl / 左程云 / labuladong / C++算法"列出所有相关笔记
4. 点过去复习具体题或教材章节
5. 想动手：滚到「经典题型」表格按 LeetCode 题号去做

---

### 场景 3：让 LLM 跨主题问答

**这是 wiki 真正的杀手级用法**。

打开 Claude Code 或 Cursor，在仓库根目录里直接问。LLM 会自动：
- 读 `algorithm-wiki/wiki/index.md` 找入口
- 用 `python algorithm-wiki/scripts/search.py "关键词"` 搜资料
- 必要时回查 `课程/`、`题目/`
- 给出带引用的回答

直接抄 [Prompt 速查](#prompt-速查可直接复制) 那一节里的任何一个开问。

---

### 场景 4：从外部抓新素材

**3 种来源，3 个去处**：

| 来源 | 去哪 |
|---|---|
| 灵神视频笔记 / 博客文章 | `algorithm-wiki/raw/articles/yyyy-mm-dd-作者-标题.md` |
| 论文 PDF | `algorithm-wiki/raw/papers/` |
| 自己整理的新专题 | `课程/灵茶山艾府 + 代码随想录/<新专题>.md` |

**抓取后**：

```bash
# 登记
python algorithm-wiki/scripts/ingest.py raw/articles/xxx.md

# 让交叉链接刷新（自动把课程/题目/题单的相关笔记接到 wiki 末尾）
python algorithm-wiki/scripts/cross_link.py
```

让 LLM 把它编译进 wiki 用 [Prompt 2](#2-把外部素材编译进-wiki)。

---

### 场景 5：每周维护

每周日（或大量新增后）跑一次：

```bash
cd "/Users/ha/Documents/study/左程云算法"

# 1. 健康检查（断链 / 孤立 / 缺 frontmatter）
python algorithm-wiki/scripts/lint.py --write-report

# 2. 刷新 wiki 文章末尾的 Vault 交叉链
python algorithm-wiki/scripts/cross_link.py

# 3. 提交 git
git add -A
git status   # 先看一眼要提交什么
git commit -m "weekly: 整理 + 新增 N 道题"
git push
```

---

## Prompt 速查（可直接复制）

> 在 Claude Code / Cursor 里直接粘下面的 prompt，按需替换 `{占位符}`。
> 假设当前目录是 Vault 根 `左程云算法/`。

### 1. 把新题链入 wiki

```text
我刚做完 LC {题号}（{题名}），写到 题目/{文件名}.md。
请按 algorithm-wiki/WORKFLOW.md 的协议：
1. 看一下 frontmatter 里 covers 字段列出的概念
2. 在 algorithm-wiki/wiki/ 中找到对应的文章
3. 在「关联条目」段（或新建「经典题型」表）加上这道题的链接
4. 完成后给我一个简短报告：更新了哪些文件
```

### 2. 把外部素材编译进 wiki

```text
请按 algorithm-wiki/WORKFLOW.md 的"ingest" 协议，
把 algorithm-wiki/raw/articles/{文件名}.md 编译进 wiki：
1. 识别 2~10 个相关概念
2. 已有概念 → 在合适小节追加新事实，参考资料段加来源
3. 应当新建的概念 → 用文章模板新建
4. 每个新条目至少 2 个反向 [[link]]
5. 完成后给报告
```

### 3. 跨主题对比（最常用）

```text
基于 algorithm-wiki/wiki/，请帮我对比 {主题A} vs {主题B}：
- 本质区别
- 各自适用场景（信号词）
- 各配 3 道代表题（如果我 Vault 里已经做过的优先列出）
- 输出到 algorithm-wiki/outputs/reports/2026-MM-DD-{topic}.md
```

例：
- 单调栈 vs 单调队列
- 回溯 vs DP
- BFS vs DFS
- 二分查找 vs 二分答案
- 0-1 背包 vs 完全背包

### 4. 出复习路线

```text
我距离面试还有 {N} 天，目标公司 {公司}，弱项 {弱项}。
请基于 algorithm-wiki/wiki/maps/algorithm-overview.md 和我的题单
（题单/Hot100.md、题单/Blind75.md），给我出一份：
- 每天 1-3 道题
- 涵盖薄弱概念
- 按"概念 → 模板 → 经典题 → 变种"顺序
输出到 algorithm-wiki/outputs/reports/2026-MM-DD-{N}day-plan.md
```

### 5. 综述某主题

```text
请基于 algorithm-wiki/wiki/ + 课程/ 综合下列资源，
给我写一份「{主题}全景综述」：
- 概念 + 模板 + 复杂度
- 信号词（什么时候想到它）
- 5 道由易到难的代表题
- 易错点 + 面试加分点
输出到 algorithm-wiki/outputs/reports/2026-MM-DD-{主题}-survey.md
```

### 6. 出模拟题

```text
请基于 algorithm-wiki 给我出 {N} 道结合 {概念A} + {概念B} 的
{难度} 难度面试题，要求：
- 题目原创但风格类似 LeetCode
- 给出测试用例
- 不要给答案（先让我自己做）
输出到 algorithm-wiki/outputs/reports/2026-MM-DD-mock-{topic}.md
```

### 7. 体检并修复

```text
请：
1. 跑 python algorithm-wiki/scripts/lint.py --write-report
2. 看 outputs/reports/lint-yyyy-mm-dd.md
3. 自动修复"明显问题"：
   - 加缺失的 frontmatter
   - 修明显笔误的断链
   - 给孤立条目找一个合理位置链入
4. "不明显的问题"列出来征求我意见
```

### 8. 把题单大文件拆成单题（破坏性，先备份）

```text
请把 题单/Hot100.md 按题号拆成 题目/0XXX-题名.md 单题文件：
- 保留每道题的 frontmatter
- 在 题单/Hot100.md 原位置改成索引（每题一行，[[]]指过去）
- 不要丢任何内容
- 完成前先 git status 确认有干净的 commit 可回退
```

### 9. 生成 Marp 幻灯片复盘

```text
请基于 algorithm-wiki/wiki/{某文章}.md 生成一份 Marp 幻灯片，
风格参考 algorithm-wiki/outputs/slides/2026-05-03-binary-search-template.md，
输出到 algorithm-wiki/outputs/slides/2026-MM-DD-{topic}.md
```

### 10. 找出我"久不复习"的概念

```text
请扫描 algorithm-wiki/wiki/ 所有文章的 frontmatter `updated` 字段，
列出 30 天内没更新过的条目，按主题分组，建议下次复习重点。
```

---

## 脚本速查

| 命令 | 用途 | 何时跑 |
|---|---|---|
| `python algorithm-wiki/scripts/ingest.py <path>` | 登记新素材到 MEMORY.md | 每次新增 raw/ 或 题目/ 文件后 |
| `python algorithm-wiki/scripts/search.py "关键词"` | 全文搜索 wiki | 想找资料时 |
| `python algorithm-wiki/scripts/search.py "关键词" --include-raw --json` | 加上原始素材 + JSON 输出（给 LLM 用） | LLM 调用 |
| `python algorithm-wiki/scripts/lint.py` | 体检 | 每周 / 大改后 |
| `python algorithm-wiki/scripts/lint.py --write-report` | 体检 + 写报告 | 同上 |
| `python algorithm-wiki/scripts/cross_link.py` | 刷新 wiki 末尾的 Vault 交叉链 | 课程/ 或 题目/ 新增后 |
| `python algorithm-wiki/scripts/compile.py <raw_path>` | 生成"编译素材"的 LLM prompt | 需要给非 Claude Code 的 LLM 用时 |

---

## 七个重要约定

1. **wiki/ 是 LLM 的领地**——你不要手写。想加内容就让 LLM 改、或写到 `outputs/reports/` 里再让 LLM 回流。
2. **课程/ 是只读快照**——保留作者原貌；想批注用 `> 编者按：…` 引用块。
3. **题目/ 是你的主战场**——每道新题一个文件，frontmatter 里写 `covers: [概念1, 概念2]`。
4. **题单/ 大文件先不动**——等真的觉得 Hot100.md 太长想拆，再用 [Prompt 8](#8-把题单大文件拆成单题破坏性先备份)。
5. **`cross_link.py` 是幂等的**——重跑会替换旧的「📂 我 Vault 里的相关笔记」段，不会重复堆积。
6. **Obsidian wikilink 按 basename 解析**——`[[链表]]` 永远能跳，不管文件移到哪个文件夹。所以**别用** `[[../../课程/.../链表]]` 这种长路径写法。
7. **git 是安全网**——每次大改前 `git status`；想回滚 `git checkout -- <文件>` 或 `git reset --hard HEAD`。

---

## 推荐节奏

### 日常（每天 30 分钟）

- 刷 1-3 题 → 写到 `题目/` → 跑 `ingest.py`

### 每周（周日 30 分钟）

- 跑 `cross_link.py` + `lint.py`
- `git add -A && git commit && git push`

### 月度（每月 1 小时）

- 让 LLM 出一份「上个月做过的题概念分布」报告 → 看哪些主题做太少
- 用 [Prompt 4](#4-出复习路线) 出下个月学习计划

### 面试前 1 个月

- 用 [Prompt 4](#4-出复习路线) 出 30 天计划
- 每个弱项主题用 [Prompt 5](#5-综述某主题) 让 LLM 写综述

### 面试前 1 周

- 用 [Prompt 6](#6-出模拟题) 每天出 3-5 道模拟题
- Mock interview 时口述思路

### 面试前 1 天

- 看 `algorithm-wiki/wiki/maps/algorithm-overview.md` 全景图
- 把所有"信号词"过一遍（"听到 X 就想到 Y"）

---

## 进阶玩法

| 想做的事 | 怎么做 |
|---|---|
| **可视化做题进度** | 装 Obsidian Dataview 插件，写查询：`TABLE difficulty, solved FROM "题目" WHERE solved = true` |
| **导出某主题题单** | 用 [Prompt 5](#5-综述某主题) 让 LLM 生成主题题单 markdown |
| **同步多设备** | git push 到 GitHub 私有 repo，电脑 / iPad（Obsidian Sync 或 Working Copy）拉 |
| **离线移动端** | iOS / Android 的 Obsidian app + git 同步 |
| **生成 Anki 卡片** | 让 LLM 把 wiki 文章转成 Anki 格式（用 prompt） |
| **Hooks 自动化** | 配 Claude Code 的 stop hook，每次会话结束自动跑 lint |
| **微调本地模型** | 长远：用 wiki 内容做合成数据，微调一个本地小模型让它"内化"你的笔记 |

---

## FAQ

**Q：我直接编辑了 wiki/ 里的某篇文章，会有问题吗？**

A：技术上没问题，但下一次让 LLM 编译素材时它可能会按自己理解再改。建议你想加内容就**告诉 LLM 加**——这样长期一致性更好。

**Q：cross_link.py 会不会把不相关的文件链进去？**

A：会有少量误匹配（按文件名关键词模糊匹配），但很容易看出来。要修正只需调 `scripts/cross_link.py` 里 `ARTICLES` 的关键词列表，重跑即可。

**Q：题目/ 的命名一定要 4 位题号吗？**

A：推荐但不强制。统一格式让排序和搜索更方便。如果是非 LeetCode 题（如剑指 Offer / 牛客），可以用 `JZ-XX-题名.md` / `NK-XX-题名.md`。

**Q：算法笔记/ 文件夹去哪了？**

A：`算法笔记/zuoAlgorithm/` 整个被搬到了 `课程/左程云/`。原文件 `算法笔记/算法笔记新手班.md` 改名搬到了 `课程/左程云/新手班.md`。

**Q：之前用 Obsidian 写过的某些 wikilink 现在指向旧路径，怎么办？**

A：Obsidian 主要按 basename 解析，所以 `[[链表]]` `[[Hot100]]` 等仍然能跳。只有用了 `[[../../旧路径/xxx]]` 这种长路径写法的链接会失效——可以用 Obsidian 的"搜索并替换"批量改，或者干脆改成短 basename 形式。

**Q：lint 报"缺参考/关联段：5"是什么意思？**

A：5 个文件没有「参考资料」或「关联条目」段。一般是 `index.md`、`MEMORY.md`、`WORKFLOW.md` 这类**元文件**——它们本来就不需要这两段。可以忽略。

**Q：我的 git 突然乱了，怎么办？**

A：千万别 `git reset --hard` 在没看清楚的情况下。先：
```bash
git status
git stash      # 暂存当前修改
git log --oneline -10   # 看最近提交
```
再决定下一步。如果搞不定就来问我，**不要乱 reset**。

---

## 一键打开（Mac 用户可以加到 ~/.zshrc）

```bash
alias algo='cd "/Users/ha/Documents/study/左程云算法"'
alias algo-lint='cd "/Users/ha/Documents/study/左程云算法" && python3 algorithm-wiki/scripts/lint.py'
alias algo-link='cd "/Users/ha/Documents/study/左程云算法" && python3 algorithm-wiki/scripts/cross_link.py'
alias algo-search='python3 "/Users/ha/Documents/study/左程云算法/algorithm-wiki/scripts/search.py"'
```

之后在任何地方：
```bash
algo                    # 切到 vault
algo-lint               # 体检
algo-link               # 刷新交叉链
algo-search "单调栈"    # 全文搜
```

---

**祝刷题顺利、面试通过 🚀**
