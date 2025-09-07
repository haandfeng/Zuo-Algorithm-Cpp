
# [114. Flatten Binary Tree to Linked List](https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/)
参考 

https://leetcode.cn/problems/flatten-binary-tree-to-linked-list/solutions/2992172/liang-chong-fang-fa-tou-cha-fa-fen-zhi-p-h9bg/

[[Hot100#[114. 二叉树展开为链表](https //leetcode.cn/problems/flatten-binary-tree-to-linked-list/)]]
# [240. Search a 2D Matrix II](https://leetcode.cn/problems/search-a-2d-matrix-ii/)
[[Hot100#[240. 搜索二维矩阵 II](https //leetcode.cn/problems/search-a-2d-matrix-ii/)]]
二分
# [3. Longest Substring Without Repeating Characters](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

滑动窗口
[[滑动窗口(同向双指针)#[3. 无重复字符的最长子串](https //leetcode.cn/problems/longest-substring-without-repeating-characters/)]]
# [198. House Robber](https://leetcode.cn/problems/house-robber/)

[[动态规划#[198. 打家劫舍](https //leetcode.cn/problems/house-robber/)]]

# [42. Trapping Rain Water](https://leetcode.cn/problems/trapping-rain-water/)

# ==[146. LRU Cache](https://leetcode.cn/problems/lru-cache/)

# ==[207. Course Schedule](https://leetcode.cn/problems/course-schedule/)

图论：给你一个有向图，判断图中是否有环。

# [1490.克隆N叉树](https://leetcode.cn/problems/clone-n-ary-tree/description/)


# [200. Number of Islands](https://leetcode.cn/problems/number-of-islands/)

# ==[会议室](https://zhuanlan.zhihu.com/p/690215895)
[[贪心#[435. 无重叠区间](https //leetcode.cn/problems/non-overlapping-intervals/)]]

# [210. Course Schedule II](https://leetcode.cn/problems/course-schedule-ii/)


# [300. Longest Increasing Subsequence](https://leetcode.cn/problems/longest-increasing-subsequence/)

[[动态规划#[300. 最长递增子序列](https //leetcode.cn/problems/longest-increasing-subsequence/)]]
# [68. Text Justification](https://leetcode.cn/problems/text-justification/)
抄了一题
https://leetcode.cn/problems/text-justification/submissions/659262360/


# [15. 3Sum](https://leetcode.cn/problems/3sum/)
[[双向双指针#[15. 三数之和](https //leetcode.cn/problems/3sum/)]]

# ==[695. Max Area of Island](https://leetcode.cn/problems/max-area-of-island/)

# [69. Sqrt(x)](https://leetcode.cn/problems/sqrtx/)
复习二分


```python
class Solution:  
    def mySqrt(self, x: int) -> int:  
        left, right = 0, x + 1  
        # left**2 <= x, right**2>x  
        while left + 1 < right:  
            m = (left + right) // 2  
            if m * m <= x:  
                left = m  
            else:  
                right = m  
        return left
```



# ==[827. Making A Large Island](https://leetcode.cn/problems/making-a-large-island/)


# [54. Spiral Matrix](https://leetcode.cn/problems/spiral-matrix/)
[[双向双指针#[59. 螺旋矩阵 II](https //leetcode.cn/problems/spiral-matrix-ii/)]]

# [5. Longest Palindromic Substring](https://leetcode.cn/problems/longest-palindromic-substring/)
参考 [[动态规划#[516. 最长回文子序列](https //leetcode.cn/problems/longest-palindromic-subsequence/)]] 解法类似

```python
class Solution:  
    def longestPalindrome(self, s: str) -> str:  
        n = len(s)  
        dp = [[0] * n for _ in range(n)]  
        maxV = (0,0,0)  
        for i in range(n-1,-1,-1):  
            for j in range(i,n):  
                 if s[i] == s[j]:  
                    if j-i <= 1:  
                        dp[i][j] = j-i+1  
                        if maxV[0] < dp[i][j]:  
                            maxV = (j-i+1,i,j)  
                    elif dp[i+1][j-1] > 0:  
                        dp[i][j] = dp[i+1][j-1]+2  
                        if maxV[0] < dp[i][j]:  
                            maxV = (dp[i][j],i,j)  
        return s[maxV[1]:maxV[2]+1]
```
# ==[678. Valid Parenthesis String](https://leetcode.cn/problems/valid-parenthesis-string/)


# [128. Longest Consecutive Sequence](https://leetcode.cn/problems/longest-consecutive-sequence/)


# ==[347. Top K Frequent Elements](https://leetcode.cn/problems/top-k-frequent-elements/)
主要是Python heap的用法 


# [53. Maximum Subarray](https://leetcode.cn/problems/maximum-subarray/)
[[动态规划#[53. 最大子序和](https //leetcode.cn/problems/maximum-subarray/)]]




# [2468. Split Message Based on Limit](https://leetcode.cn/problems/split-message-based-on-limit/)


# [399. Evaluate Division](https://leetcode.cn/problems/evaluate-division/)



# [322. Coin Change](https://leetcode.cn/problems/coin-change/)
[[动态规划#[322. 零钱兑换](https //leetcode.cn/problems/coin-change/)]]
和518对比着来做

# [71. Simplify Path](https://leetcode.cn/problems/simplify-path/)


# [33. Search in Rotated Sorted Array](https://leetcode.cn/problems/search-in-rotated-sorted-array/)



# [121. Best Time to Buy and Sell Stock](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

 