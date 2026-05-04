---
leetcode: 560
title: Subarray Sum Equals K
url: https://leetcode.com/problems/subarray-sum-equals-k/
difficulty: medium
tags:
  - prefix-sum
covers:
  - prefix-sum
sources:
  - 灵茶山艾府和代码随想录/前缀和
date_split: 2026-05-05
solved: true
---

# 560. Subarray Sum Equals K

[LeetCode 链接](https://leetcode.com/problems/subarray-sum-equals-k/)

## 来源：灵茶山艾府和代码随想录 / 前缀和

[[Hot100#[560. 和为 K 的子数组](https //leetcode.cn/problems/subarray-sum-equals-k/)]]

我认为主要思路是: 明显是要求一个区间的和->前缀和。但具体要求哪个区间并不知道，可能是多个区间。但数学公式是不变的，是 si - sj = k 。 所以思考成两数之和
