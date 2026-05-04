---
title: 滑动窗口
aliases: [Sliding Window, 滑窗, 同向双指针]
tags: [technique]
created: 2026-05-03
updated: 2026-05-03
---

# 滑动窗口

> **TL;DR**：滑窗 = **同向双指针 + 一个动态窗口状态**。右指针扩展、左指针收缩。能把"枚举所有子段"的 O(N²) 降到 **O(N)**。专治"连续子数组 / 子串 + 满足某约束 + 求最长 / 最短 / 计数"。

## 通用模板

```cpp
int slidingWindow(vector<int>& nums, int target) {
    int l = 0, ans = 0;
    State window;                        // 窗口当前的状态（hash / sum / cnt 等）
    for (int r = 0; r < (int)nums.size(); ++r) {
        // 1. 把 a[r] 加入窗口
        window.add(nums[r]);

        // 2. 收缩：当窗口不合法时移出 a[l]
        while (窗口不合法) {
            window.remove(nums[l]);
            l++;
        }

        // 3. 此时 [l, r] 是合法窗口，更新答案
        ans = max(ans, r - l + 1);
    }
    return ans;
}
```

**核心**：右指针**只前进**，左指针**也只前进**，每个元素被加入 / 移出窗口各一次 → **总 O(N)**。

## 三类经典题

### 1. 不定长窗口求"最长"

```cpp
// LC 3 无重复字符的最长子串
int lengthOfLongestSubstring(string s) {
    unordered_map<char, int> cnt;
    int l = 0, ans = 0;
    for (int r = 0; r < (int)s.size(); ++r) {
        cnt[s[r]]++;
        while (cnt[s[r]] > 1) {       // 不合法：当前字符已出现过
            cnt[s[l]]--;
            l++;
        }
        ans = max(ans, r - l + 1);
    }
    return ans;
}
```

### 2. 不定长窗口求"最短"

```cpp
// LC 209 长度最小的子数组（和 ≥ target）
int minSubArrayLen(int target, vector<int>& nums) {
    int l = 0, sum = 0, ans = INT_MAX;
    for (int r = 0; r < (int)nums.size(); ++r) {
        sum += nums[r];
        while (sum >= target) {            // 满足时收缩，找最短
            ans = min(ans, r - l + 1);
            sum -= nums[l];
            l++;
        }
    }
    return ans == INT_MAX ? 0 : ans;
}
```

注意"最短"题型的收缩条件和"最长"反过来。

### 3. 定长窗口

```cpp
// LC 643 子数组的最大平均数 I
double findMaxAverage(vector<int>& a, int k) {
    int n = a.size();
    int sum = 0;
    for (int i = 0; i < k; ++i) sum += a[i];
    int best = sum;
    for (int i = k; i < n; ++i) {
        sum += a[i] - a[i - k];
        best = max(best, sum);
    }
    return best / (double)k;
}
```

定长窗口不需要复杂的收缩逻辑——直接前进时进 1 出 1。

## 何时想到滑窗

信号词识别：

| 信号 | 形态 |
|---|---|
| "连续子数组 / 子串" | ✅ 强信号 |
| "满足……的最长 / 最短" | ✅ 强信号 |
| "和 / 字符出现次数 / 不同字符数" | ✅ 强信号 |
| "恰好 K 个不同元素" | ✅ 滑窗 + 容斥（"最多 K"-"最多 K-1"） |
| "可以删除 K 个字符 / 反转 K 个" | ✅ 等价于"窗口内某条件 ≤ K" |

如果序列**不连续**（要选不相邻元素）→ 用 [[dp]] 不是滑窗。

## "最多 K 个" / "恰好 K 个" 套路

"恰好 K 个"通常拆成"最多 K 个" - "最多 K - 1 个"：

```cpp
int subarraysWithKDifferent(vector<int>& nums, int K) {
    return atMost(nums, K) - atMost(nums, K - 1);
}

int atMost(vector<int>& nums, int K) {
    unordered_map<int,int> cnt;
    int l = 0, ans = 0, distinct = 0;
    for (int r = 0; r < (int)nums.size(); ++r) {
        if (cnt[nums[r]]++ == 0) distinct++;
        while (distinct > K) {
            if (--cnt[nums[l]] == 0) distinct--;
            l++;
        }
        ans += r - l + 1;             // 以 r 结尾的合法子数组数
    }
    return ans;
}
```

LC 992 K 个不同整数的子数组用的就是这招。

## 经典题型

| 题 | 类型 |
|---|---|
| 无重复字符的最长子串 LC 3 | 不定长 / 最长 |
| 长度最小的子数组 LC 209 | 不定长 / 最短 |
| 最小覆盖子串 LC 76 | 不定长 / 最短 + hash |
| 字符串的排列 LC 567 | 定长 + 字母 cnt 比较 |
| 找到字符串中所有字母异位词 LC 438 | 定长 + 字母 cnt |
| 滑动窗口最大值 LC 239 | 定长 + [[monotonic-queue\|单调队列]] |
| 子数组的最大平均数 LC 643 | 定长 |
| 子数组中和的最大值 LC 1004 替换后最长连续 1 | 不定长 / 最长 |
| 至多包含 K 个不同字符的最长子串 LC 340 | 不定长 / 最长 |
| K 个不同整数的子数组 LC 992 | "最多 K" - "最多 K-1" |
| 滑动窗口中位数 LC 480 | 双堆 / multiset |

## 与其它技巧的关系

- 同向双指针 = 滑窗 = "右扩 + 左缩"
- [[monotonic-queue|单调队列]] 是滑窗求**最值**的工具
- [[prefix-sum|前缀和]] 是处理**任意子段**的工具，**子段 + 单调约束**才转滑窗
- [[binary-search-answer|二分答案]] 偶尔可以替代某些滑窗题

## 易错点

- 收缩条件写反：求"最长"要收缩到合法、求"最短"要收缩到不合法的临界
- 状态更新位置：`add` 在循环开头还是结尾、`remove` 在 `l++` 之前还是之后？要严格对应
- 求"恰好 K 个"直接写很难——拆成"最多" - "最多 K-1"
- 字符 cnt 用 `unordered_map` 容易 TLE，能用 `int cnt[26]` 就用数组
- "右指针超出后才更新答案" vs "每个 r 都更新"：定长 vs 不定长不同

## 📂 我 Vault 里的相关笔记

> 自动扫描 Vault 中含「滑动窗口」相关关键词（滑动窗口, 滑窗）的笔记，按来源分组。

### 灵茶山艾府和代码随想录
- [[滑动窗口(同向双指针)|课程/灵茶山艾府和代码随想录/滑动窗口(同向双指针).md]]

### 左程云
- [[滑动窗口 & 单调栈|课程/左程云/002. 知识点总结/20. 算法知识点/滑动窗口 & 单调栈.md]]
- [[滑动窗口|课程/左程云/002. 知识点总结/20. 算法知识点/滑动窗口.md]]
- [[滑动窗口模板|课程/左程云/010. 代码模板/滑动窗口模板.md]]
- [[第1节 滑动窗口和单调栈|课程/左程云/100. 算法课程/21. 训练营第1期/第1节 滑动窗口和单调栈.md]]
- [[滑动窗口中位数|课程/左程云/200. ★算法题目汇总★/★算法面试题汇总/滑动窗口中位数.md]]
- [[滑动窗口内的最大值|课程/左程云/200. ★算法题目汇总★/★算法面试题汇总/滑动窗口内的最大值.md]]

### labuladong
- [[滑动窗口|课程/labuladong/核心刷题框架/滑动窗口.md]]

## 参考资料

- [[../../课程/灵茶山艾府和代码随想录/滑动窗口(同向双指针)]] — 你的滑窗专题
- [[../../课程/labuladong/核心刷题框架]] — labuladong 滑窗框架

## 关联条目

- 上层：[[algorithm-overview]]
- 兄弟：[[two-pointers]] · [[monotonic-queue]] · [[prefix-sum]]
- 数据结构：[[hash-table]] · [[queue]]
