
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



# [80. 删除有序数组中的重复项 II](https://leetcode.cn/problems/remove-duplicates-from-sorted-array-ii/)

```python

```



# [169. 多数元素](https://leetcode.cn/problems/majority-element/) 