https://leetcode.com/problem-list/rab78cw1/
# [1. 两数之和](https://leetcode.cn/problems/two-sum/)

顺便根据灵神的题单，延展出几道题

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        idx = {}  # 创建一个空哈希表（字典）
        for j, x in enumerate(nums):  # x=nums[j]
            if target - x in idx:  # 在左边找 nums[i]，满足 nums[i]+x=target
                return [idx[target - x], j]  # 返回两个数的下标
            idx[x] = j  # 保存 nums[j] 和 j
```
## [167. 两数之和 II - 输入有序数组](https://leetcode.cn/problems/two-sum-ii-input-array-is-sorted/)

双指针法，因为已经排好序了，从左从右看，求和，小了左指针往右走，大了右指针往左走。



```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left = 0
        right = len(numbers) - 1
        while True:
            s = numbers[left] + numbers[right]
            if s == target:
                return [left + 1, right + 1]  # 题目要求下标从 1 开始
            if s > target:
                right -= 1
            else:
                left += 1
```


## [15. 三数之和](https://leetcode.cn/problems/3sum/)

需要注意，三数之和不是两数之和的变形。是上一题的变形，为方便双指针以及跳过相同元素，先把 nums 排序。排序是合理的，因为答案的数据量是Cn3 是 n^2 空间复杂度

灵神还有额外的优化，一些简单的数学

优化一：如果 nums[i] 与后面最小的两个数相加 nums[i]+nums[i+1]+nums[i+2]>0，那么后面不可能存在三数之和等于 0，break 外层循环。

优化二：如果 nums[i] 与后面最大的两个数相加 nums[i]+nums[n−2]+nums[n−1]<0，那么内层循环不可能存在三数之和等于 0，但继续枚举，nums[i] 可以变大，所以后面还有机会找到三数之和等于 0，continue 外层循环。



```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        ans = []
        n = len(nums)
        for i in range(n - 2):
            x = nums[i]
            if i > 0 and x == nums[i - 1]:  # 跳过重复数字
                continue
            if x + nums[i + 1] + nums[i + 2] > 0:  # 优化一
                break
            if x + nums[-2] + nums[-1] < 0:  # 优化二
                continue
            j = i + 1
            k = n - 1
            while j < k:
                s = x + nums[j] + nums[k]
                if s > 0:
                    k -= 1
                elif s < 0:
                    j += 1
                else:  # 三数之和为 0
                    ans.append([x, nums[j], nums[k]])
                    j += 1
                    while j < k and nums[j] == nums[j - 1]:  # 跳过重复数字
                        j += 1
                    k -= 1
                    while k > j and nums[k] == nums[k + 1]:  # 跳过重复数字
                        k -= 1
        return ans
```

## [18. 四数之和](https://leetcode.cn/problems/4sum/)

有优化思路，但懒得想了
```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        ans = []
        for i in range(0,len(nums)-3):
            if i >= 1 and nums[i] == nums[i-1]:
                continue
            for j in range(i+1,len(nums)-2):
                if j >= i+2 and nums[j] == nums[j-1]:
                    continue
                left = j + 1
                right = len(nums) - 1
                real_target = target - nums[i] -nums[j]
                while left < right:
                    if nums[left] + nums[right] == real_target:
                        ans.append([nums[i],nums[j],nums[left],nums[right]])
                        left += 1
                        while left < right and nums[left-1] == nums[left]:
                            left+=1
                    elif nums[left] + nums[right] < real_target:
                        left += 1
                    else:
                        right -=1
        return ans   
```
## [1512. 好数对的数目](https://leetcode.cn/problems/number-of-good-pairs/)
1分题，不会做得反思。
```python
class Solution:
    def numIdenticalPairs(self, nums: List[int]) -> int:
        ans = 0
        cnt = defaultdict(int)
        for x in nums:  # x = nums[j]
            # 此时 cnt[x] 表示之前遍历过的 x 的个数，加到 ans 中
            # 如果先执行 cnt[x] += 1，再执行 ans += cnt[x]，就把 i=j 这种情况也统计进来了，算出的答案会偏大
            ans += cnt[x]
            cnt[x] += 1
        return ans
```

## [219. 存在重复元素 II](https://leetcode.cn/problems/contains-duplicate-ii/)

注意下标的更新时机
```python
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        d = {}
        for i in range(0,len(nums)):
            if nums[i] not in d:
                d[nums[i]] = i
            else:
                if i - d[nums[i]] <=k:
                    return True
                else:
                    d[nums[i]] = i
        return False
```

## [1010. 总持续时间可被 60 整除的歌曲](https://leetcode.cn/problems/pairs-of-songs-with-total-durations-divisible-by-60/)

```python
from typing import List

class Solution:
    def numPairsDivisibleBy60(self, time: List[int]) -> int:
        d = {}
        ans = 0
        for t in time:
            r = t % 60
            # 边界处理：如果余数为0，对应的补数也是0
            complement = (60 - r) % 60
            
            if complement in d:
                ans += d[complement]
            
            d[r] = d.get(r, 0) + 1
            
        return ans
```
## [2748. 美丽下标对的数目](https://leetcode.cn/problems/number-of-beautiful-pairs/)

枚举 x=nums[j]，我们需要知道有多少个 nums[i]，满足 i<j 且 nums[i] 的最高位与 xmod10 互质。

需要直接枚举 nums[i] 吗？有没有更快的做法？

由于 nums[i] 的最高位在 [1,9] 中，我们可以在遍历数组的同时，统计最高位的出现次数，这样就只需枚举 [1,9] 中的与 xmod10 互质的数，把对应的出现次数加到答案中。

具体算法如下：

```python
class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        ans = 0
        cnt = [0] * 10
        for x in nums:
            for y, c in enumerate(cnt):
                if c and gcd(y, x % 10) == 1:
                    ans += c
            while x >= 10: 
                x //= 10
            cnt[x] += 1  # 统计最高位的出现次数
        return ans
```