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



# [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        ans = left = 0
        window = set()  # 维护从下标 left 到下标 right 的字符
        for right, c in enumerate(s):
            # 如果窗口内已经包含 c，那么再加入一个 c 会导致窗口内有重复元素
            # 所以要在加入 c 之前，先移出窗口内的 c
            while c in window:  # 窗口内有 c
                window.remove(s[left])
                left += 1  # 缩小窗口
            window.add(c)  # 加入 c
            ans = max(ans, right - left + 1)  # 更新窗口长度最大值
        return ans
```
# [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)
区间DP, 注意，如果s[i]!=s[j]，不需要处理，因为此时不是回文子串。dp\[i]\[j]的定义是以i，j结尾的子串的最大回文子串长度。区分好子串和子序列
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
# [133. 克隆图](https://leetcode.cn/problems/clone-graph/)

简单的遍历，用哈希表记录访问过的节点
```python
class Solution(object):

    def __init__(self):
        self.visited = {}

    def cloneGraph(self, node):
        """
        :type node: Node
        :rtype: Node
        """
        if not node:
            return node

        # 如果该节点已经被访问过了，则直接从哈希表中取出对应的克隆节点返回
        if node in self.visited:
            return self.visited[node]

        # 克隆节点，注意到为了深拷贝我们不会克隆它的邻居的列表
        clone_node = Node(node.val, [])

        # 哈希表存储
        self.visited[node] = clone_node

        # 遍历该节点的邻居并更新克隆节点的邻居列表
        if node.neighbors:
            clone_node.neighbors = [self.cloneGraph(n) for n in node.neighbors]

        return clone_node
```


# [383. 赎金信](https://leetcode.cn/problems/ransom-note/)

一个哈希表的事情
```python
class Solution:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        if len(ransomNote) > len(magazine):
            return False
        cnt = defaultdict(int)
        for c in magazine:
            cnt[c] += 1
        for c in ransomNote:
            cnt[c] -= 1
        return all(c >= 0 for c in cnt.values())
```


# [8. 字符串转换整数 (atoi)](https://leetcode.cn/problems/string-to-integer-atoi/)
状态的转换，面试挺常见的，可以多看看，自己再写一次
```python
class Solution:
    def myAtoi(self, s: str) -> int:
        s = s.strip()                      # 删除首尾空格
        if not s: return 0                   # 字符串为空则直接返回
        res, i, sign = 0, 1, 1
        int_max, int_min, bndry = 2 ** 31 - 1, -2 ** 31, 2 ** 31 // 10
        if s[0] == '-': sign = -1            # 保存负号
        elif s[0] != '+': i = 0              # 若无符号位，则需从 i = 0 开始数字拼接
        for c in s[i:]:
            if not '0' <= c <= '9' : break     # 遇到非数字的字符则跳出
            if res > bndry or res == bndry and c > '7': return int_max if sign == 1 else int_min # 数字越界处理
            res = 10 * res + ord(c) - ord('0') # 数字拼接
        return sign * res
```

# [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)

一开始想的是单调栈，单调栈是找比当前元素大下一个更大的元素。这里虽然也是要到所有元素下一个更大的元素。但其实在这里，我们只需要找到比当前元素下一个更大的数就好了，用双指针解决
这里没用到上面的思路，还可以优化的。


```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height)-1
        maxArea = 0
        while left < right:
            maxArea = max(maxArea, (right - left) * min(height[right], height[left]))
            if height[left] <= height[right]:
                left+=1
            else:
                right-=1
        return maxArea
```
# [139. 单词拆分](https://leetcode.cn/problems/word-break/)
有思路还是要写出来，这样才知道自己是对的还是错的。
我的思路是回溯，但回溯理论上是从0开始的，如果从n开始，很多都可以写成dp（不严谨，只是遇到的情况）。

所以如果回溯觉得从0开始不好想，可以想象从n开始回溯。回溯完之后，可以再考虑，回溯递归是同一个参数重复进入，是否结果会不会一样，如果不会，那么可以计划搜索

考虑到整个递归过程中有大量重复递归调用（递归入参相同）。由于递归函数没有副作用，同样的入参无论计算多少次，算出来的结果都是一样的，因此可以用记忆化搜索来优化：
```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        max_len = max(map(len, wordDict))  # 用于限制下面 j 的循环次数
        words = set(wordDict)  # 便于快速判断 s[j:i] in words

        n = len(s)
        f = [True] + [False] * n
        for i in range(1, n + 1):
            for j in range(i - 1, max(i - max_len - 1, -1), -1):
                if f[j] and s[j:i] in words:
                    f[i] = True
                    break
        return f[n]
```

# [141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/)

如果有环，会一直循环下去，直到相遇。没环的话会走到None
```python
class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow, fast = head, head
        while fast != None and fast.next != None:
            slow = slow.next
            fast = fast.next.next
            if fast == slow:
                return True
        return False
```

## [142. 环形链表 II](https://leetcode.cn/problems/linked-list-cycle-ii/)
经典套路，fast先和slow相遇。slow和head再同时走直到相遇，具体推导可以看灵神，这里用的是灵神的代码，感觉他的更简洁
``
```python
class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if fast is slow:  # 相遇
                while slow is not head:  # 再走 a 步
                    slow = slow.next
                    head = head.next
                return slow
        return None
```
# [876. 链表的中间结点](https://leetcode.cn/problems/middle-of-the-linked-list/)

快慢指针运用
```python
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow, fast = head, head
        while fast and fast.next:
            fast = fast.next.next
            slow = slow.next
        return slow
```


# [15. 3Sum](https://leetcode.com/problems/3sum/)
[[Grind75#[15. 三数之和](https //leetcode.cn/problems/3sum/)]]



# [994. 腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/)

注意写的见解点，还有最后一次的不会让橘子变腐烂的特殊情况
```python
from collections import deque
from typing import List

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        m = len(grid)
        n = len(grid[0])
        queue = deque()
        flag = True  # 是否一开始就没有新鲜橘子
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    queue.append((i, j))
                elif grid[i][j] == 1:
                    flag = False

        ans = 0
        if flag:
            return 0

        while queue:
            new_Rotting_number = len(queue)
            for _ in range(new_Rotting_number):
                x, y = queue.popleft()
                # 只看上下左右四个方向
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1:
                        grid[nx][ny] = 2
                        queue.append((nx, ny))
            ans += 1

        # 检查是否还有新鲜橘子
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 1:
                    return -1

        # 这里 ans 会多加 1 层（最后一层没有再感染新的），减去 1
        return ans - 1
```


# [17. 电话号码的字母组合](https://leetcode.cn/problems/letter-combinations-of-a-phone-number/)

```python
class Solution:
    letters = ["", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]

    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        self.path = []
        ans = []

        def backTrace(i: int) -> None:
            if i == len(digits):
                ans.append("".join(self.path))
                return
            dig = int(digits[i]) -1
            for ch in self.letters[dig]:
                self.path.append(ch)
                backTrace(i + 1)
                self.path.pop()

        backTrace(0)
        return ans
```

# [146. LRU 缓存](https://leetcode.cn/problems/lru-cache/)


注意怎么做成一个环状的双向链表，要有一个dummy node一开始prev 和 next指向自己。这样就可以抽象了。 
insert节点的时候，如果一开始只有 dummy node，那么即dummy node的下一个节点是自己，下一个节点的prev也是自己。所以在只有dummy node并往头节点插入的时候，插入的节点next节点是dummy node 的next就是dummy node。并且next节点的prev要指向自己，所以next(dummy node)的prev指向了当前插入的节点，所以形成了环
![[Pasted image 20251018143732.png]]



同理在删除节点的时候，原本需要，node的前一个节点的next指向node的下一个节点。node的下一个节点的prev指向node的前一个节点。
但如果只有dummy的另外一个节点的情况下，删除当前节点。当前节点的下一个节点的prev会指向当前节点的前一个节点，即dummy。其他同理，这样子可以实现逻辑自洽的添加删除操作。
![[Pasted image 20251018144157.png]]


```python
class DoubleLinkedNode:
    def __init__(self, key, value):
        self.val = value
        self.key = key
        self.next = None
        self.prev = None
class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.dummyNode = DoubleLinkedNode(-1,-1)
        self.dummyNode.next = self.dummyNode
        self.dummyNode.prev = self.dummyNode
        self.cache = {}
    def putFirst(self, key: int) -> None:
        node= self.cache[key]
        node.next = self.dummyNode.next
        node.prev = self.dummyNode
        self.dummyNode.next = node
        node.next.prev = node
    def remove(self, key: int):
        node = self.cache[key]
        node.prev.next = node.next
        node.next.prev = node.prev
    def get(self, key: int) -> int:
        if key in self.cache:
            self.remove(key)
            self.putFirst(key)
            return self.cache[key].val
        else:
            return -1
    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache[key].val = value
            self.remove(key)
            self.putFirst(key)
        else:
            if self.size == self.capacity:
                self.size-=1
                temp = self.dummyNode.prev.key
                self.remove(temp)
                self.cache.pop(temp)
            node = DoubleLinkedNode(key,value)
            self.cache[key] = node
            self.putFirst(key)
            self.size+=1
```


## [460. LFU 缓存](https://leetcode.cn/problems/lfu-cache/)


### 我的解法
主要难点有两个
1. 想到为每个freq维护写一个双链表，并且理解什么时候更新最小的freq
2. 模块化的思想。如果这题写的不够模块化，很容易乱，导致出现未知bug
```python
class DoubleLinkedNode:
    def __init__(self, key, value):
        self.key = key
        self.val = value
        self.next = None
        self.prev = None
        self.freq = 0


class LFUCache:
    def __init__(self, capacity: int):
        self.size = 0
        self.capacity = capacity
        self.counter = {}
        self.cache = {}
        self.minCount = 1

    def remove(self, key: int):
        node = self.cache[key]
        node.prev.next = node.next
        node.next.prev = node.prev

    def addFront(self, key: int):
        node = self.cache[key]
        dummyNode = self.counter[node.freq]
        node.prev = dummyNode
        node.next = dummyNode.next
        node.prev.next = node
        node.next.prev = node

    def moveNode(self, key: int):
        node = self.cache[key]
        if node.freq not in self.counter:
            # create dummy node
            dummyNode = DoubleLinkedNode(-1, -1)
            dummyNode.prev = dummyNode
            dummyNode.next = dummyNode
            self.counter[node.freq] = dummyNode
            # add the node to the front
            self.addFront(key)
        else:
            self.addFront(key)

    def popCounter(self, freq: int) -> bool:
        if self.counter[freq].next == self.counter[freq]:
            self.counter.pop(freq)
            return True
        return False

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            node.freq += 1
            self.remove(key)
            if self.popCounter(node.freq - 1):
                if self.minCount == node.freq - 1:
                    self.minCount = node.freq
            self.moveNode(key)
            return node.val
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            node = self.cache[key]
            node.freq += 1
            node.val = value
            self.remove(key)            
            if self.popCounter(node.freq - 1):
                if self.minCount == node.freq - 1:
                    self.minCount = node.freq
            self.moveNode(key)
        else:
            if self.size == self.capacity:
                self.size -= 1
                deleteKey = self.counter[self.minCount].prev.key
                self.remove(deleteKey)
                if self.popCounter(self.minCount):
                    self.minCount = 1
                self.cache.pop(deleteKey)
            node = DoubleLinkedNode(key, value)
            node.freq = 1
            self.cache[key] = node
            self.moveNode(key)
            self.minCount = 1
            self.size += 1
```


### 0x3f

![460-2-c.png](https://pic.leetcode.cn/1695621293-JySfYQ-460-2-c.png)


```python
class Node:
    # 提高访问属性的速度，并节省内存
    __slots__ = 'prev', 'next', 'key', 'value', 'freq'

    def __init__(self, key=0, val=0):
        self.key = key
        self.value = val
        self.freq = 1  #  新书只读了一次

class LFUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.key_to_node = {}
        def new_list() -> Node:
            dummy = Node()  # 哨兵节点
            dummy.prev = dummy
            dummy.next = dummy
            return dummy
        self.freq_to_dummy = defaultdict(new_list)

    def get_node(self, key: int) -> Optional[Node]:
        if key not in self.key_to_node:  # 没有这本书
            return None
        node = self.key_to_node[key]  # 有这本书
        self.remove(node)  # 把这本书抽出来
        dummy = self.freq_to_dummy[node.freq]
        if dummy.prev == dummy:  # 抽出来后，这摞书是空的
            del self.freq_to_dummy[node.freq]  # 移除空链表
            if self.min_freq == node.freq:  # 这摞书是最左边的
                self.min_freq += 1
        node.freq += 1  # 看书次数 +1
        self.push_front(self.freq_to_dummy[node.freq], node)  # 放在右边这摞书的最上面
        return node

    def get(self, key: int) -> int:
        node = self.get_node(key)
        return node.value if node else -1

    def put(self, key: int, value: int) -> None:
        node = self.get_node(key)
        if node:  # 有这本书
            node.value = value  # 更新 value
            return
        if len(self.key_to_node) == self.capacity:  # 书太多了
            dummy = self.freq_to_dummy[self.min_freq]
            back_node = dummy.prev  # 最左边那摞书的最下面的书
            del self.key_to_node[back_node.key]
            self.remove(back_node)  # 移除
            if dummy.prev == dummy:  # 这摞书是空的
                del self.freq_to_dummy[self.min_freq]  # 移除空链表
        self.key_to_node[key] = node = Node(key, value)  # 新书
        self.push_front(self.freq_to_dummy[1], node)  # 放在「看过 1 次」的最上面
        self.min_freq = 1

    # 删除一个节点（抽出一本书）
    def remove(self, x: Node) -> None:
        x.prev.next = x.next
        x.next.prev = x.prev

    # 在链表头添加一个节点（把一本书放到最上面）
    def push_front(self, dummy: Node, x: Node) -> None:
        x.prev = dummy
        x.next = dummy.next
        x.prev.next = x
        x.next.prev = x
```

# [20. 有效的括号](https://leetcode.cn/problems/valid-parentheses/)

注意stack为空的情况
```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        for ch in s:
            if ch == '(':
                stack.append(')')
            elif ch == '[':
                stack.append((']'))
            elif ch == '{':
                stack.append('}')
            else:
                if stack and ch == stack[-1]:
                    stack.pop()
                else:
                    return False
        return True if not stack else False
```


# [21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)
记得链表的哨兵技巧，特别是涉及到第一个节点的问题
```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        if not list1:
            return list2
        if not list2:
            return list1
        dummy =  ListNode(-1)
        head = dummy
        while list1 and list2:
            if list1.val <= list2.val:
                head.next = list1
                list1 = list1.next
                head = head.next
            else:
                head.next = list2
                list2 = list2.next
                head = head.next
        if list1:
            head.next = list1
        if list2:
            head.next = list2
        return dummy.next
```


# [150. 逆波兰表达式求值](https://leetcode.cn/problems/evaluate-reverse-polish-notation/)
注意python运算的坑点和如何避免

![[Pasted image 20251018173229.png]]
```python
class Solution(object):
    def evalRPN(self, tokens):
        stack = []
        for token in tokens:
            try:
                stack.append(int(token))
            except:
                num2 = stack.pop()
                num1 = stack.pop()
                stack.append(self.evaluate(num1, num2, token))
        return stack[0]

    def evaluate(self, num1, num2, op):
        if op == "+":
            return num1 + num2
        elif op == "-":
            return num1 - num2
        elif op == "*":
            return num1 * num2
        elif op == "/":
            return int(num1 / float(num2))
```




# [23. 合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)
为什么能想到堆，我觉得是我希望找到k个数里的最小的数字，并且找到之后，要再找下一个数字，需要一直对比，然后找最小，所以这个时候用堆。时间负责度一定要O(n), 这里建堆的时候件事logk * n  符合要求，所以可以用。

这里的id是为了防止node.val一样的时候，堆无法比较
```python
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        heap = []
        for node in lists:
            if node:
                # 元组结构: (比较值, 唯一序号, 节点)
                heapq.heappush(heap, (node.val, id(node), node))

        dummy = ListNode(0)
        cur = dummy

        while heap:
            _, _, node = heapq.heappop(heap)
            cur.next = node
            cur = cur.next
            if node.next:
                heapq.heappush(heap, (node.next.val, id(node.next), node.next))
        return dummy.next
```
# [278. 第一个错误的版本](https://leetcode.cn/problems/first-bad-version/)
二分法，建议先看下面的基础二分
```python

class Solution:
    def firstBadVersion(self, n: int) -> int:
        left, right = 1, n # 左闭右闭指的是，要看的数的范围，从0到len-1
        while left <= right:
            # nums < left is good, nums > right is bad
            mid = (left+right) // 2
            if isBadVersion(mid):
                right = mid - 1
            else:
                left = mid + 1
        return right+1
```
# [409. 最长回文串](https://leetcode.cn/problems/longest-palindrome/)
贪心的思路，把贪心的情况想全就好了，没什么难的
```python
class Solution:
    def longestPalindrome(self, s: str) -> int:
        # Dictionary to store frequency of occurrence of each character
        frequency_map = {}
        # Count frequencies
        for c in s:
            frequency_map[c] = frequency_map.get(c, 0) + 1

        res = 0
        has_odd_frequency = False
        for freq in frequency_map.values():
            # Check if the frequency is even
            if (freq % 2) == 0:
                res += freq
            else:
                # If the frequency is odd, one occurrence of the
                # character will remain without a match
                res += freq - 1
                has_odd_frequency = True

        # If has_odd_frequency is true, we have at least one unmatched
        # character to make the center of an odd length palindrome.
        if has_odd_frequency:
            return res + 1

        return res
```
# [704. 二分查找](https://leetcode.cn/problems/binary-search/)

详细看注释，重写二分的时候发现自己还是遇到了一些问题，下面的注释是一些思考
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums)-1 # 左闭右闭指的是，要看的数的范围，从0到len-1
        while left <= right: #所以只有错开的时候，才是全部看完了
            mid = (left + right) // 2
            if nums[mid] < target:
                left = mid + 1 # left的左边的所有数字都是< target的
            else:
                right = mid - 1 #right的右边所有数字都是 > target的
        # 注意这里返回的right + 1 时 >= target的第一个数，而不是一定是target所以还要做判断
        return right + 1 if right + 1 < len(nums) and nums[right + 1] == target else -1
```


# [155. 最小栈](https://leetcode.cn/problems/min-stack/)

我的思路是维护两个栈，一个是正常栈。一个是单调栈。
我对这个题的理解是：弹出一个数后，找到下一个最小值。 例如 -2 0 -3。 0 是绝对不可能成为最小值的，这是找下一个最小值，其实并不符合单调栈的使用方法。单调栈是找到下一个上一个或下一个更大的值。
感觉灵神的会更好 https://leetcode.cn/problems/min-stack/solutions/2974438/ben-zhi-shi-wei-hu-qian-zhui-zui-xiao-zh-x0g8/ 维护前缀最小值

但我的思路是：我们只需要知道 下一个最小值 -2 0 -3。 0绝对不可能成为下一个最小值，因为0>栈底，所以0不入栈，-3 < -2所以是新的最小值，-3入栈
```python

class MinStack:

    def __init__(self):
        self.stack = []
        self.monoStack = []
    def push(self, val: int) -> None:
        self.stack.append(val)
        if self.monoStack and val <= self.monoStack[-1]:
            self.monoStack.append(val)
        elif not self.monoStack:
            self.monoStack.append(val)
    def pop(self) -> None:
        val = self.stack.pop()
        if val == self.monoStack[-1]:
            self.monoStack.pop()
    def top(self) -> int:
        return self.stack[-1] if self.stack else -1
    def getMin(self) -> int:
        return self.monoStack[-1] if self.monoStack else -1
```



# [981. 基于时间的键值存储](https://leetcode.cn/problems/time-based-key-value-store/)


# [542. 01 矩阵](https://leetcode.cn/problems/01-matrix/)


在一个图中，能从一个点出发求这种最短距离的方法很容易想到就是 BFS，

这是我的代码，从一个0开始出发，找下一个一。但要注意入队条件，==只要1更新了，就要重新入队，因为1变小了，会影响前后左右的1。==
```python
class Solution:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m, n= len(mat),  len(mat[0])
        res = [ [float('inf')] * n for _ in range(m)]
        firstZero = []
        for i in range(0, m):
            for j in range (0, n):
                if mat[i][j] == 0:
                    firstZero.append(i)
                    firstZero.append(j)
                    res[i][j] = 0
                    break
            if firstZero:
                break
        queue = deque()
        queue.append(firstZero)
        while queue:
            dirction = [[-1,0],[1,0],[0,-1],[0,1]]
            coordinate = queue.popleft()
            for nx, ny in dirction:
                x, y = coordinate[0] + nx, coordinate[1] + ny
                if 0 <= x < m and 0 <= y < n:
                    if mat[x][y] == 0:
                        if res[x][y] == 0:
                            continue
                        res[x][y] = 0
                        queue.append([x,y])
                    elif mat[x][y] == 1:
                        if res[x][y] < res[coordinate[0]][coordinate[1]] + 1:
                            queue.append([x, y])
                            res[x][y] = res[coordinate[0]][coordinate[1]] + 1
        return res
```



也可以从所有0开始，找到离这些0最近的1的点。一层一层的向外拓展，这个时候第一次遇到1的点，肯定是他的最小值


```python
from collections import deque
from typing import List

class Solution:
    def updateMatrix(self, matrix: List[List[int]]) -> List[List[int]]:
        # 获取矩阵的行数和列数
        m, n = len(matrix), len(matrix[0])

        # 初始化结果矩阵，所有值设为 0（后续会在 BFS 中更新）
        dist = [[0] * n for _ in range(m)]

        # 找出所有值为 0 的格子位置，存成一个列表 [(i1, j1), (i2, j2), ...]
        # 这些位置会作为 BFS 的起点（多源 BFS）
        zeroes_pos = [(i, j) for i in range(m) for j in range(n) if matrix[i][j] == 0]

        # 将所有 0 的位置放入队列中，表示这些点的距离为 0
        q = deque(zeroes_pos)

        # 用一个集合记录已经访问过的格子（避免重复入队）
        seen = set(zeroes_pos)

        # 方向数组：上、下、左、右
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # BFS 主循环：从所有 0 同时向外扩散
        while q:
            i, j = q.popleft()  # 当前出队的格子坐标

            # 遍历 4 个方向的邻居格子
            for di, dj in directions:
                ni, nj = i + di, j + dj
                # 检查是否越界，且该邻居尚未访问
                if 0 <= ni < m and 0 <= nj < n and (ni, nj) not in seen:
                    # 邻居的距离 = 当前格子的距离 + 1
                    dist[ni][nj] = dist[i][j] + 1
                    # 入队继续向外扩散
                    q.append((ni, nj))
                    # 标记为已访问
                    seen.add((ni, nj))

        return dist
```