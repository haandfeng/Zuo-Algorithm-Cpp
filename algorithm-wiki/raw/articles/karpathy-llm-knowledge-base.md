---
source: https://x.com/karpathy
author: Andrej Karpathy
captured: 2026-05-03
tags: [methodology, knowledge-base, raw]
---

# Karpathy: LLM Knowledge Bases

> 编者按：本帖是这个 Wiki 的方法论起点。下面是原文（英文），中文整理与"算法领域定制"已写在 Wiki 的 [`README.md`](../../README.md) 和 [`WORKFLOW.md`](../../WORKFLOW.md)。

Something I'm finding very useful recently: using LLMs to build personal knowledge bases for various topics of research interest. In this way, a large fraction of my recent token throughput is going less into manipulating code, and more into manipulating knowledge (stored as markdown and images). The latest LLMs are quite good at it. So:

**Data ingest:** I index source documents (articles, papers, repos, datasets, images, etc.) into a `raw/` directory, then I use an LLM to incrementally "compile" a wiki, which is just a collection of `.md` files in a directory structure. The wiki includes summaries of all the data in `raw/`, backlinks, and then it categorizes data into concepts, writes articles for them, and links them all. To convert web articles into `.md` files I like to use the Obsidian Web Clipper extension, and then I also use a hotkey to download all the related images to local so that my LLM can easily reference them.

**IDE:** I use Obsidian as the IDE "frontend" where I can view the raw data, the the compiled wiki, and the derived visualizations. Important to note that the LLM writes and maintains all of the data of the wiki, I rarely touch it directly. I've played with a few Obsidian plugins to render and view data in other ways (e.g. Marp for slides).

**Q&A:** Where things get interesting is that once your wiki is big enough (e.g. mine on some recent research is ~100 articles and ~400K words), you can ask your LLM agent all kinds of complex questions against the wiki, and it will go off, research the answers, etc. I thought I had to reach for fancy RAG, but the LLM has been pretty good about auto-maintaining index files and brief summaries of all the documents and it reads all the important related data fairly easily at this ~small scale.

**Output:** Instead of getting answers in text/terminal, I like to have it render markdown files for me, or slide shows (Marp format), or matplotlib images, all of which I then view again in Obsidian. You can imagine many other visual output formats depending on the query. Often, I end up "filing" the outputs back into the wiki to enhance it for further queries. So my own explorations and queries always "add up" in the knowledge base.

**Linting:** I've run some LLM "health checks" over the wiki to e.g. find inconsistent data, impute missing data (with web searchers), find interesting connections for new article candidates, etc., to incrementally clean up the wiki and enhance its overall data integrity. The LLMs are quite good at suggesting further questions to ask and look into.

**Extra tools:** I find myself developing additional tools to process the data, e.g. I vibe coded a small and naive search engine over the wiki, which I both use directly (in a web ui), but more often I want to hand it off to an LLM via CLI as a tool for larger queries.

**Further explorations:** As the repo grows, the natural desire is to also think about synthetic data generation + finetuning to have your LLM "know" the data in its weights instead of just context windows.

**TLDR:** raw data from a given number of sources is collected, then compiled by an LLM into a `.md` wiki, then operated on by various CLIs by the LLM to do Q&A and to incrementally enhance the wiki, and all of it viewable in Obsidian. You rarely ever write or edit the wiki manually, it's the domain of the LLM. I think there is room here for an incredible new product instead of a hacky collection of scripts.

---

## 落地到「算法学习」场景的关键映射

| 原帖 | 我们的算法 Wiki |
|---|---|
| `raw/` 是博客 / 论文 / repo | `raw/` 主要是 **指回 Vault 已有笔记** + 极少量外部资料 |
| Wiki 是研究领域综述 | Wiki 是**概念地图**：复杂度 / 数据结构 / 算法 / 解题技巧 / 题型 |
| Q&A 跨多篇研究文章 | Q&A 跨多个题目和专题（"哪些 Hot100 用了单调栈？"） |
| Output 是综述 / 幻灯 | Output 是**对比表 / 刷题路线图 / Marp 复盘** |
| Lint 找事实矛盾 | Lint 找**复杂度断言错误 / 模板代码错误 / 题号引用失效** |
| 进一步：合成数据 + 微调 | 进一步：**用 wiki 生成自己的练习题、自动出 mock interview 题** |
