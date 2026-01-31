
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

```python
class Solution:  
    def trap(self, height: List[int]) -> int:  
        stack = []  
        ans = 0  
        for i in range(len(height)):  
            if not stack:  
                stack.append(i)  
            else:  
                if height[i] <= height[stack[-1]]:  
                    stack.append(i)  
                else:  
                    while stack and height[i] > height[stack[-1]]:  
                        mid = stack.pop()  
                        if not stack:  
                            break  
                        left = height[stack[-1]]  
                        right = i  
                        high = min(height[right],height[left]) - height[mid]  
                        width = right-left-1  
                        ans += high * width  
                    stack.append(i)  
        return ans
```


# ==[146. LRU Cache](https://leetcode.cn/problems/lru-cache/)

# [207. Course Schedule](https://leetcode.cn/problems/course-schedule/)

[[零茶山艾府+代码随想录/图#[207. 课程表](https //leetcode.cn/problems/course-schedule/)]]

# [1490.克隆N叉树](https://leetcode.cn/problems/clone-n-ary-tree/description/)


# [200. Number of Islands](https://leetcode.cn/problems/number-of-islands/)

# ==会议室(https://zhuanlan.zhihu.com/p/690215895)

https://zhuanlan.zhihu.com/p/690215895
# [210. Course Schedule II](https://leetcode.cn/problems/course-schedule-ii/)
[[零茶山艾府+代码随想录/图#[210. 课程表 II](https //leetcode.cn/problems/course-schedule-ii/)]]

# [300. Longest Increasing Subsequence](https://leetcode.cn/problems/longest-increasing-subsequence/)

[[动态规划#[300. 最长递增子序列](https //leetcode.cn/problems/longest-increasing-subsequence/)]]
# [68. Text Justification](https://leetcode.cn/problems/text-justification/)
抄了一题
https://leetcode.cn/problems/text-justification/submissions/659262360/
不难，看注释，别想差就好了

```python
from typing import List

class Solution:
    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        res = []
        row = []
        cntL = 0   # 当前行单词总长度
        cntW = 0   # 当前行单词数

        for word in words:
            # 先尝试把这个词放进来（需要+ (cntW) 个空格位，因为已有 cntW 个词就有 cntW 个间隙最少要1空格）
            if cntL + len(word) + cntW > maxWidth:
                # 不能再放当前 word 了，先结算上一行（row 里是上一行的词）
                if cntW == 1:
                    # 只有一个词：左对齐，右边补空格
                    line = row[0] + " " * (maxWidth - cntL)
                else:
                    # 多个词：均匀分配空格
                    total_spaces = maxWidth - cntL                   # 总空格数（包含基础空格）
                    gaps = cntW - 1
                    base = total_spaces // gaps                      # 每个间隙至少的空格数
                    extra = total_spaces % gaps                      # 从左到右多出来的空格数

                    parts = []
                    for i in range(gaps):
                        parts.append(row[i])
                        # 基础空格 + 左侧优先分配的 extra
                        spaces = base + (1 if i < extra else 0)
                        parts.append(" " * spaces)
                    parts.append(row[-1])  # 最后一个词后不加空格
                    line = "".join(parts)

                res.append(line)

                # 新起一行，从当前 word 开始
                row = [word]
                cntL = len(word)
                cntW = 1
            else:
                # 放得下就加入当前行
                row.append(word)
                cntL += len(word)
                cntW += 1

        # 最后一行：左对齐，单词间一个空格，右侧补齐
        last_line = " ".join(row)
        last_line += " " * (maxWidth - len(last_line))
        res.append(last_line)

        return res
```
# [15. 3Sum](https://leetcode.cn/problems/3sum/)
[[双向双指针#[15. 三数之和](https //leetcode.cn/problems/3sum/)]]

# [695. Max Area of Island](https://leetcode.cn/problems/max-area-of-island/)

[[零茶山艾府+代码随想录/图#[695. 岛屿的最大面积](https //leetcode.cn/problems/max-area-of-island/)]]

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



# [827. Making A Large Island](https://leetcode.cn/problems/making-a-large-island/)
[[零茶山艾府+代码随想录/图#[827. 最大人工岛](https //leetcode.cn/problems/making-a-large-island/)]]

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
# [678. Valid Parenthesis String](https://leetcode.cn/problems/valid-parenthesis-string/)
需要记录index而不是数值，index包含更多信息

# [128. Longest Consecutive Sequence](https://leetcode.cn/problems/longest-consecutive-sequence/)


# ==[347. Top K Frequent Elements](https://leetcode.cn/problems/top-k-frequent-elements/)
主要是Python heap的用法 

[[Hot100#[347. 前 K 个高频元素](https //leetcode.cn/problems/top-k-frequent-elements/)]]

# [53. Maximum Subarray](https://leetcode.cn/problems/maximum-subarray/)
[[动态规划#[53. 最大子序和](https //leetcode.cn/problems/maximum-subarray/)]]




# ==[2468. Split Message Based on Limit](https://leetcode.cn/problems/split-message-based-on-limit/)


# [399. Evaluate Division](https://leetcode.cn/problems/evaluate-division/)



# ==[322. Coin Change](https://leetcode.cn/problems/coin-change/)
[[动态规划#[322. 零钱兑换](https //leetcode.cn/problems/coin-change/)]]
和518对比着来做

# [71. Simplify Path](https://leetcode.cn/problems/simplify-path/)


# [33. Search in Rotated Sorted Array](https://leetcode.cn/problems/search-in-rotated-sorted-array/)



# ==[121. Best Time to Buy and Sell Stock](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

# [560. Subarray Sum Equals K](https://leetcode.cn/problems/subarray-sum-equals-k/)

[[Hot100#[560. 和为 K 的子数组](https //leetcode.cn/problems/subarray-sum-equals-k/)]]
# [1248. Count Number of Nice Subarrays](https://leetcode.cn/problems/count-number-of-nice-subarrays/)


恰好k个
```python
from typing import List

class Solution:
    def numberOfSubarrays(self, nums: List[int], k: int) -> int:
        left = 0
        cnt = 0        # 窗口内奇数个数
        prefix = 0     # 在当前窗口 cnt==k 时可选左端点数量
        res = 0
        for x in nums:
            if x % 2 == 1:
                cnt += 1
                prefix = 0            # 新奇数出现，prefix 需重算
            while cnt == k:
                if nums[left] % 2 == 1:
                    cnt -= 1
                left += 1
                prefix += 1           # 每右移一步 left，多一种左端点
            res += prefix             # 以当前 x 为右端点的所有“恰好 k”子数组
        return res
```


# [994. Rotting Oranges](https://leetcode.cn/problems/rotting-oranges/)
图的知识
```python
def orangesRotting(self, grid: List[List[int]]) -> int:  
    dirs =[(0,1),(1,0),(0,-1),(-1,0)]  
    m = len(grid)  
    n = len(grid[0])  
    queue = deque()  
    for i in range(m):  
        for j in range(n):  
            if grid[i][j] == 2:  
                queue.append((i,j))  
    res = 0  
    while queue:  
        for i in range(len(queue)):  
            position = queue.popleft()  
            for r,c in dirs:  
                if position[0]+r < 0 or position[0]+r >=m or position[1]+c <0 or position[1]+c>=n:  
                    continue  
                if grid[position[0]+r][position[1]+c] == 1:  
                    grid[position[0] + r][position[1] + c] = 2  
                    queue.append((position[0]+r,position[1]+c))  
        res+=1  
  
    for i in range(m):  
        for j in range(n):  
            if grid[i][j] == 1:  
                return -1  
    return res-1 if res>0 else 0
```
# [125. Valid Palindrome](https://leetcode.cn/problems/valid-palindrome/)

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        i, j = 0, len(s) - 1
        while i < j:
            if not s[i].isalnum():
                i += 1
            elif not s[j].isalnum():
                j -= 1
            elif s[i].lower() == s[j].lower():
                i += 1
                j -= 1
            else:
                return False
        return True
```


# [45.Jump Game II](https://leetcode.cn/problems/jump-game-ii/)

[[贪心#[45. 跳跃游戏 II](https //leetcode.cn/problems/jump-game-ii/)]]



# [593. Valid Square](https://leetcode.cn/problems/valid-square/)
一道数学题，自己慢慢解，懒得写


# [22. Generate Parentheses](https://leetcode.cn/problems/generate-parentheses/)
[[回溯#[22. 括号生成](https //leetcode.cn/problems/generate-parentheses/)]]



# [316. Remove Duplicate Letters](https://leetcode.cn/problems/remove-duplicate-letters/)
看看答案吧，我不知道有什么套路，就一些对set和统计的运用


# ==[739. Daily Temperatures](https://leetcode.cn/problems/daily-temperatures/)
[[零茶山艾府+代码随想录/单调栈#[739. 每日温度](https //leetcode.cn/problems/daily-temperatures/)|每日温度]]

# [76. Minimum Window Substring](https://leetcode.cn/problems/minimum-window-substring/)
[[滑动窗口(同向双指针)#[76. 最小覆盖子串](https //leetcode.cn/problems/minimum-window-substring/)]]


# [1466. Reorder Routes to Make All Paths Lead to the City Zero](https://leetcode.cn/problems/reorder-routes-to-make-all-paths-lead-to-the-city-zero/)


# [47. Permutations II](https://leetcode.cn/problems/permutations-ii/)

[[回溯#[47. 全排列 II](https //leetcode.cn/problems/permutations-ii/)]]



# [2021. Brightest Position on Street](https://leetcode.com/problems/brightest-position-on-street/)

