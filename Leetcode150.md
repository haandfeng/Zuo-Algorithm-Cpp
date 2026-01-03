
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
看0x3f的思路
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