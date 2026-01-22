
# [88. 合并两个有序数组](https://leetcode.cn/problems/merge-sorted-array/)

方法二中，之所以要使用临时变量，是因为如果直接合并到数组 nums 1中，nums 1中的元素可能会在取出之前被覆盖。那么如何直接避免覆盖 nums 1中的元素呢？观察可知，nums1的后半部分是空的，可以直接覆盖而不会影响结果。因此可以指针设置为从后向前遍历，每次取两者之中的较大者放进 nums 1的最后面。


```python
class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -> None:
        """
        Do not return anything, modify nums1 in-place instead.
        """
        p1, p2 = m - 1, n - 1
        tail = m + n - 1
        while p1 >= 0 or p2 >= 0:
            if p1 == -1:
                nums1[tail] = nums2[p2]
                p2 -= 1
            elif p2 == -1:
                nums1[tail] = nums1[p1]
                p1 -= 1
            elif nums1[p1] > nums2[p2]:
                nums1[tail] = nums1[p1]
                p1 -= 1
            else:
                nums1[tail] = nums2[p2]
                p2 -= 1
            tail -= 1
```

# [26. 删除有序数组中的重复项](https://leetcode.cn/problems/remove-duplicates-from-sorted-array/)


```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        prev = nums[0]
        head = 1
        uniqueStart = 1
        while head < len(nums):
            if nums[head] != prev:
                prev = nums[head]
                temp = nums[uniqueStart]
                nums[uniqueStart] = nums[head]
                nums[head] = temp
                uniqueStart+=1
            else:
                prev = nums[head]
            head +=1
        return uniqueStart
```


0x3f的简单很多
```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        k = 1
        for i in range(1, len(nums)):
            if nums[i] != nums[i - 1]:  # nums[i] 不是重复项
                nums[k] = nums[i]  # 保留 nums[i]
                k += 1
        return k
```
# [80. 删除有序数组中的重复项 II](https://leetcode.cn/problems/remove-duplicates-from-sorted-array-ii/)
看0x3f的思路，有一个空间模拟栈，主要还是覆盖就好了，然后用一个index模拟覆盖的位置
```python
class Solution:
    def removeDuplicates(self, nums: List[int]) -> int:
        stack_size = 2  # 栈的大小，前两个元素默认保留
        for i in range(2, len(nums)):
            if nums[i] != nums[stack_size - 2]:  # 和栈顶下方的元素比较
                nums[stack_size] = nums[i]  # 入栈
                stack_size += 1
        return min(stack_size, len(nums))
```



# [169. 多数元素](https://leetcode.cn/problems/majority-element/) 

```python
class Solution:
    def majorityElement(self, nums):
        # Counter([1, 2, 2, 3, 3, 3])
        # 输出：Counter({3: 3, 2: 2, 1: 1})
        counts = collections.Counter(nums)
        # counts.keys() 获取所有不同的元素（字典的键）。
        # max(iterable, key=function) 从一个可迭代对象（如列表、字典的键等）中找出使得 function(x) 最大的那个元素。
        return max(counts.keys(), key=counts.get)
```

# [189. 轮转数组](https://leetcode.cn/problems/rotate-array/)

转多几次
```python
# 注：请勿使用切片，会产生额外空间
class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        def reverse(i: int, j: int) -> None:
            while i < j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1

        n = len(nums)
        k %= n  # 轮转 k 次等于轮转 k % n 次
        reverse(0, n - 1)
        reverse(0, k - 1)
        reverse(k, n - 1)
```


# [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
            @cache
            def dfs(n, hold):
                if n < 0:
                    return -inf if hold else 0
                if hold:
                    return max(-prices[n], dfs(n-1,True))
                return max(dfs(n-1,False), dfs(n-1,True)+prices[n])
            return dfs(len(prices)-1, False)
```


# [122. 买卖股票的最佳时机 II](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-ii/)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        @cache
        def dfs(n: int, hold: bool):
            if n < 0:
                return float('-inf') if hold else 0    
            if hold:
                return max(dfs(n-1,False)-prices[n], dfs(n-1,True))
            return max(dfs(n-1,False),dfs(n-1,True)+prices[n])
        return dfs(len(prices)-1, False)     
            
```
# [55. 跳跃游戏](https://leetcode.cn/problems/jump-game/)

```python
class Solution:
    def canJump(self, nums: List[int]) -> bool:
        mx = 0
        for i, jump in enumerate(nums):
            if i > mx:  # 无法到达 i
                return False
            mx = max(mx, i + jump)  # 从 i 最右可以跳到 i + jump
            if mx >= len(nums) - 1:  # 可以跳到 n-1
                return True
```



# [45. 跳跃游戏 II](https://leetcode.cn/problems/jump-game-ii/)

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        if len(nums) <=1:
            return 0
        mx = nums[0]
        curPos = 0
        step = 0
        while mx < len(nums) - 1:
            step+=1
            for i in range(curPos+1,mx + 1):
                if nums[i] + i > mx:
                    curPos = i
                    mx = nums[i] + i
        step+=1
        return step
```
# [274. H 指数](https://leetcode.cn/problems/h-index/)


题意： 给你一个数组，求一个最大的 h，使得数组中有至少 h 个数都大于等于 h。

本题可以做到 O(n) 时间。

设 n 为 citations 的长度，即这名研究者发表的论文数。根据题意，h 不可能超过 n，所以对于引用次数大于 n 的论文，我们在统计的时候，可以看成是引用次数等于 n 的论文。例如 n = 5，假设 h 是 5，那么无论引用次数是 6 还是 5，都满足 >= 5，所以 6 可以看成是 5，毕竟我们只需要统计有多少个数字是 >= 5 的。

所以，创建一个长度为 n + 1 的 cnt 数组，统计 min(citations[i], n) 的出现次数。

设 s 为引用次数 >= i 的论文数，我们需要算出满足 s >= i 的最大的 i。

为了快速算出有多少论文的引用次数 >= i，我们可以从 i = n 开始倒序循环，每次循环，把 cnt[i] 加到 s 中。由于我们是倒序循环的，只要 s >= 成立，此时的 i 就是满足 s >= i 的最大的 i，直接返回 i 作为答案。

例如示例 1，从 i = 5 开始：
	1.	i = 5，现在 s = 2 < i，继续循环。
	2.	i = 4，现在 s = 2 < i，继续循环。
	3.	i = 3，现在 s = 3 >= i，返回 3。

The value sk is defined as "the sum of all counts with citation ≥k" or "the number of papers having, at least, k citations". By definition of the h-index, the largest k with k≤sk
​
  is our answer.

```python
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        n = len(citations)
        cnt = [0] * (n + 1)
        for c in citations:
            cnt[min(c, n)] += 1  # 引用次数 > n，等价于引用次数为 n
        s = 0
        for i in range(n, -1, -1):  # i=0 的时候，s>=i 一定成立
            s += cnt[i]
            if s >= i:  # 说明有至少 i 篇论文的引用次数至少为 i
                return i
```
## [275. H 指数 II](https://leetcode.cn/problems/h-index-ii/)
还是要理解h，感觉没理解好
有序直接二分, n是一共有n篇论文
i 是 有 i+1篇文章 <= citations[i], 所以一共有n-i篇论文 >= citation[i]

我们要找的是：
	•	h = n - i
	•	条件：citations[i] >= h 也就是 citations[i] >= n - i

所以我们想找的是第一次“符合条件”的位置（最小 i 使得 citations[i] >= h）。->这样子就是有h篇文章，大于h引用


```python
class Solution:
    def hIndex(self, citations: List[int]) -> int:
        if not citations or citations[-1] == 0:
            return 0
        # 左边（不满足）：citations[i] < n - i（说明 h 太大/引用太小，需要往右找更大的 citations）
        # 右边（满足）：citations[i] >= n - i（说明已经够了，尝试往左找更小 i）
        left = 0 
        right = len(citations) - 1
        # 1 2 100

        while left <= right:
            mid = (left + right) // 2
            if len(citations) - mid > citations[mid]:
                left = mid + 1
            elif len(citations) - mid < citations[mid]:
                right = mid - 1
            else:
                return len(citations) - mid
        return len(citations) - left
```

# [380. O(1) 时间插入、删除和获取随机元素](https://leetcode.cn/problems/insert-delete-getrandom-o1/)



# [238. 除了自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/)


# [134. 加油站](https://leetcode.cn/problems/gas-station/)

核心思路：“已经在谷底了，怎么走都是向上。
找到最小的差

# [135. 分发糖果](https://leetcode.cn/problems/candy/)


# [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/)

学习一下怎么描述怎么解法
```python
class Solution:
    def trap(self, height: List[int]) -> int:
        stack = []
        res = 0
        for i, h in enumerate(height):
            while stack and height[stack[-1]] < h:
                mid = stack.pop()
                if stack:
                    left = stack[-1]
                    res += (min(height[left],height[i]) - height[mid]) * (i-left-1)
            stack.append(i)
        return res
```



# [28. 找出字符串中第一个匹配项的下标](https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/)
https://leetcode.cn/problems/find-the-index-of-the-first-occurrence-in-a-string/solutions/575568/shua-chuan-lc-shuang-bai-po-su-jie-fa-km-tb86/?envType=study-plan-v2&envId=top-interview-150

看这个，学习KMP算法 + 卡玛
主要是模式串的next数组O（m）的时间复杂度创建

# [135. 分发糖果](https://leetcode.cn/problems/candy/)


# [68. 文本左右对齐](https://leetcode.cn/problems/text-justification/)
```python
from typing import List

class Solution:
    def retLine(self, line: List[str], cnt: int, maxWidth: int, lastLine: bool) -> str:
        s = ""
        if not lastLine:
            if len(line) > 1:
                spaceNum = (maxWidth - cnt) // (len(line) - 1)
                remainSpace = (maxWidth - cnt) % (len(line) - 1)

                for i, word in enumerate(line):
                    if i != len(line) - 1:  # 不是最后一个
                        s += word + " " * spaceNum
                        if remainSpace > 0:
                            s += " "
                            remainSpace -= 1
                    else:
                        s += word
            else:
                s = line[0] + " " * (maxWidth - len(line[0]))
        else:
            for i, word in enumerate(line):
                if i != len(line) - 1:
                    s += word + " "
                else:
                    s += word + " " * (maxWidth - cnt - (len(line) - 1))
        return s

    def fullJustify(self, words: List[str], maxWidth: int) -> List[str]:
        cnt = 0
        line = []
        res = []

        for word in words:
            if cnt + len(word) + len(line) > maxWidth:
                res.append(self.retLine(line, cnt, maxWidth, False))
                cnt = 0
                line = []
            cnt += len(word)
            line.append(word)

        res.append(self.retLine(line, cnt, maxWidth, True))
        return res
```







# [123. 买卖股票的最佳时机 III](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iii/)
```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        dp = [[float('-inf')] * 4 for _ in range (len(prices))]
        # 0(1卖出)，2（3卖出）不持有， 1 ，3 持有
        dp[0][0] = 0
        dp[0][1] = -prices[0]
        dp[0][2] = 0
        dp[0][3] = -prices[0]
        for i in range(1,len(prices)):
            dp[i][0] = max(dp[i-1][0], dp[i-1][1]+prices[i])
            dp[i][1] = max(dp[i-1][1], -prices[i])
            dp[i][2] = max(dp[i-1][2], dp[i-1][3]+prices[i])
            dp[i][3] = max(dp[i-1][3], dp[i-1][0]-prices[i])
        return dp[len(prices)-1][2]
```


# [188. 买卖股票的最佳时机 IV](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock-iv/)
```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        f = [[[-inf] * 2 for _ in range(k + 2)] for _ in range(n + 1)]
        for j in range(1, k + 2):
            f[0][j][0] = 0
        for i, p in enumerate(prices):
            for j in range(1, k + 2):
                f[i + 1][j][0] = max(f[i][j][0], f[i][j][1] + p)
                f[i + 1][j][1] = max(f[i][j][1], f[i][j - 1][0] - p)
        return f[-1][-1][0]
```



