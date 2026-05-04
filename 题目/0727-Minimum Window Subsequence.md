---
leetcode: 727
title: Minimum Window Subsequence
url: https://leetcode.com/problems/minimum-window-subsequence/
difficulty: 
tags: []
covers: []
sources:
  - 题单/Blind75
date_split: 2026-05-05
solved: true
---

# 727. Minimum Window Subsequence

[LeetCode 链接](https://leetcode.com/problems/minimum-window-subsequence/)

## 来源：题单 / Blind75

Tiktok 变式，使用动态规划，dp\[i][j] = 表示的是 S【j-1】 T【i-1】他们匹配的的起始位置是 dp\[i][j] - 1 
```java
class Solution {
    public String minWindow(String S, String T) {
        int m = T.length(), n = S.length();
        int[][] dp = new int[m + 1][n + 1];
        for (int j = 0; j <= n; j++) {
            dp[0][j] = j + 1;
        }
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                if (T.charAt(i - 1) == S.charAt(j - 1)) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = dp[i][j - 1];
                }
            }
        }

        int start = 0, len = n + 1;
        for (int j = 1; j <= n; j++) {
            if (dp[m][j] != 0) {
                if (j - dp[m][j] + 1 < len) {
                    start = dp[m][j] - 1;
                    len = j - dp[m][j] + 1;
                }
            }
        }
        return len == n + 1 ? "" : S.substring(start, start + len);
    }
}
```
