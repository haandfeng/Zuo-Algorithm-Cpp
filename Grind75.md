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

## 方法1
```python
class TimeMap:
    def __init__(self):
        # key -> list of (timestamp, value)
        self.m = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.m[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        pairs = self.m[key]
        # bisect_left(a, x) 在第一个等于 x 的位置前插入  即bisect_left：返回 “第一个 ≥ x” 的位置；
        # bisect_left(a, x) 在最后一个等于 x 的位置前插入 即bisect_right：返回 “第一个 > x” 的位置。
        # chr(127) 生成一个“比所有 value 都大的虚拟值”)
        # bisect_right 返回第一个 > (timestamp, dummy_value) 的位置
        i = bisect.bisect_right(pairs, (timestamp, chr(127)))
        if i > 0:
            return pairs[i - 1][1]
        return ""
```

## 方法2
会超时，但这个存的方法不错。把搜索改成二分查找应该可以过
```python
class TimeMap:
    def __init__(self):
        self.key_time_map = {}

    def set(self, key: str, value: str, timestamp: int) -> None:
        # If the 'key' does not exist in dictionary.
        if not key in self.key_time_map:
            self.key_time_map[key] = {}
            
        # Store '(timestamp, value)' pair in 'key' bucket.
        self.key_time_map[key][timestamp] = value
        

    def get(self, key: str, timestamp: int) -> str:
        # If the 'key' does not exist in dictionary we will return empty string.
        if not key in self.key_time_map:
            return ""
        
        # Iterate on time from 'timestamp' to '1'.
        for curr_time in reversed(range(1, timestamp + 1)):
            # If a value for current time is stored in key's bucket we return the value.
            if curr_time in self.key_time_map[key]:
                return self.key_time_map[key][curr_time]
            
        # Otherwise no time <= timestamp was stored in key's bucket.
        return ""
```
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


# [543. 二叉树的直径](https://leetcode.cn/problems/diameter-of-binary-tree/)

相关题目看0x3f的课
```python
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:
        self.maxR = 0
        def dfs(root: Optional[TreeNode]) -> int:
            if root == None:
                return -1
            left = dfs(root.left) + 1
            right = dfs(root.right) + 1
            self.maxR = max(self.maxR,left+right)
            return max(left,right)
        dfs(root)
        return self.maxR
```


## [416. 分割等和子集](https://leetcode.cn/problems/partition-equal-subset-sum/)
做过很多遍了

```python
class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        numsSum  = sum(nums)
        if numsSum % 2 == 1 :
            return False
        dp = [0] * (numsSum // 2 + 1)
        for num in nums:
            for j in range(numsSum // 2, num-1, -1):
                dp[j] = max(dp[j],dp[j-num] + num)
        if dp[-1] == numsSum // 2 :
            return  True
        return False
```


# [33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)
## 前置题: [153. 寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/solution/by-endlesscheng-owgd/)
这里对left和right的定义是一样的，但是我们只要找到旋转的点就好了，旋转的点就是最小值，所以mid只需要和nums 0比较
```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left,right = -1, len(nums)
        if nums[0] <= nums[-1]: return nums[0]
        while left+1<right:
            mid = (left+right)//2
            if nums[mid] >= nums[0]:
                left = mid
            else:
                right = mid
        return nums[right]

```


难点是把边界条件写清楚
```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # left's left is target's left, right's right is target or target's right
        left, right = 0, len(nums) -1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return mid
            if nums[0] <= nums[-1]:
                if nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            else:
                if nums[mid] > target and target >= nums[0]:
                    right = mid -1
                elif nums[mid] < nums[0] and target >= nums[0]:
                    right = mid -1
                elif nums[mid] > target and nums[mid] < nums[0] and target < nums[0] :
                    right = mid -1
                else:
                    left = mid + 1
        return left if  left < len(nums) and nums[left] == target else -1
```






# [39. 组合总和](https://leetcode.cn/problems/combination-sum/)
思路大致差不多，但没办法区分，什么时候用回溯，什么时候用dp
```python
class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        ans = []
        path = []

        def dfs(i: int, left: int) -> None:
            if left == 0:
                # 找到一个合法组合
                ans.append(path.copy())
                return

            if i == len(candidates) or left < 0:
                return

            # 不选
            dfs(i + 1, left)

            # 选
            path.append(candidates[i])
            dfs(i, left - candidates[i])
            path.pop()  # 恢复现场

        dfs(0, target)
        return ans
```

## [40. 组合总和 II](https://leetcode.cn/problems/combination-sum-ii/)
解决重复的问题
```python
class Solution:
    def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
        ans = []
        path = []
        candidates.sort()
        def dfs(i: int, left: int) -> None:
            if left == 0:
                # 找到一个合法组合
                ans.append(path.copy())
                return

            if i == len(candidates) or left < 0:
                return

            # 选
            path.append(candidates[i])
            dfs(i + 1, left - candidates[i])
            path.pop()  # 恢复现场

            # 不选
            while i < len(candidates) - 1 and candidates[i] == candidates[i+1]:
                i+=1
            dfs(i + 1, left)


        dfs(0, target)
        return ans
```


## [216. 组合总和 III](https://leetcode.cn/problems/combination-sum-iii/)

两种方法

和前面1 和 2 一样的方法

```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        ans = []
        path = []

        def dfs(i: int, left_sum: int) -> None:
            d = k - len(path)  # 还要选 d 个数
            if left_sum < 0 or left_sum > (i * 2 - d + 1) * d // 2:  # 剪枝
                return
            if d == 0:  # 找到一个合法组合
                ans.append(path.copy())
                return

            # 不选 i
            if i > d:
                dfs(i - 1, left_sum)

            # 选 i
            path.append(i)
            dfs(i - 1, left_sum - i)
            path.pop()

        dfs(9, n)
        return ans
```



## [377. 组合总和 Ⅳ](https://leetcode.cn/problems/combination-sum-iv/)

```python
class Solution:
    def combinationSum4(self, nums: List[int], target: int) -> int:
        f = [1] + [0] * target
        for i in range(1, target + 1):
            f[i] = sum(f[i - x] for x in nums if x <= i)
        return f[target]
```
# [295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/)


```python
import heapq

class MedianFinder:
    def __init__(self):
        # leftMaxHeap 保存较小的一半，作为大根堆（通过存负数）
        self.leftMaxHeap = []   # store negatives, so heapq is min-heap of negatives => max-heap behavior
        # rightMinHeap 保存较大的一半，作为普通小根堆
        self.rightMinHeap = []  # store positives, min-heap

    def addNum(self, num: int) -> None:
        # 首先决定把 num 放到哪个堆：
        # 如果 left 为空或 num <= 左堆的最大值（即 -leftMaxHeap[0]），放左堆；否则放右堆
        if not self.leftMaxHeap or num <= -self.leftMaxHeap[0]:
            heapq.heappush(self.leftMaxHeap, -num)
        else:
            heapq.heappush(self.rightMinHeap, num)

        # 之后保持平衡：left 允许比 right 多 1，否则调整
        if len(self.leftMaxHeap) > len(self.rightMinHeap) + 1:
            # left 太大，移动一个到 right
            val = -heapq.heappop(self.leftMaxHeap)
            heapq.heappush(self.rightMinHeap, val)
        elif len(self.rightMinHeap) > len(self.leftMaxHeap):
            # right 太大，移动一个到 left
            val = heapq.heappop(self.rightMinHeap)
            heapq.heappush(self.leftMaxHeap, -val)

    def findMedian(self) -> float:
        # sizes
        leftSize = len(self.leftMaxHeap)
        rightSize = len(self.rightMinHeap)

        if leftSize == rightSize:
            if leftSize == 0:
                return 0.0  # 或抛出异常，视你需求
            # 两堆顶的平均
            return (-self.leftMaxHeap[0] + self.rightMinHeap[0]) / 2.0
        else:
            # left 比 right 多 1 时中位数为 left 的最大值
            return float(-self.leftMaxHeap[0])
```


#  [169. 多数元素](https://leetcode.cn/problems/majority-element/)

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




# [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/)

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        stack = []
        ans = 0
        for i in range(len(height)):
            while stack and height[i] > height[stack[-1]]:
                right  = i
                mid = stack.pop()
                if stack:
                    left = stack[-1]
                else:
                    break
                ans += (right - left - 1) * (min(height[left],height[right]) - height[mid])
            stack.append(i)
        return ans
```


# [297. 二叉树的序列化与反序列化](https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/)

没什么难度，就是bfs，但是代码没写好，写得不够简洁，虽然能写出来就好了

```python
from collections import deque
from typing import Optional, List


class Codec:
    def serialize(self, root: Optional[TreeNode]) -> str:
        """Encodes a tree to a single string."""
        if root is None:
            return "None"

        q = deque([root])
        ans: List[str] = []

        while q:
            node = q.popleft()
            if node is None:
                ans.append("None")
            else:
                ans.append(str(node.val))
                q.append(node.left)
                q.append(node.right)

        # 去掉末尾多余的 "None"，让输出更短
        while ans and ans[-1] == "None":
            ans.pop()

        return ",".join(ans)

    def deserialize(self, data: str) -> Optional[TreeNode]:
        """Decodes your encoded data to tree."""
        if not data or data == "None":
            return None

        vals = data.split(",")
        root = TreeNode(int(vals[0]))
        q = deque([root])
        i = 1  # 指向下一个要处理的值

        while q and i < len(vals):
            node = q.popleft()

            # 左子
            if i < len(vals) and vals[i] != "None":
                node.left = TreeNode(int(vals[i]))
                q.append(node.left)
            i += 1

            # 右子
            if i < len(vals) and vals[i] != "None":
                node.right = TreeNode(int(vals[i]))
                q.append(node.right)
            i += 1

        return root
```

# [46. 全排列](https://leetcode.cn/problems/permutations/)

```python
from typing import List
class Solution(object):
    def permute(self, nums: List[int]) -> List[List[int]]:
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        n = len(nums)
        ans = []
        path = [0] * n
        on_path = [False] * n
        def dfs(i:int) -> None:
            if i == n:
                ans.append(path.copy())
                return
            for j,on in enumerate(on_path):
                if not on:
                    path[i] = nums[j]
                    on_path[j] = True
                    dfs(i+1)
                    on_path[j] = False
        dfs(0)
        return ans
```

## [47. 全排列 II](https://leetcode.cn/problems/permutations-ii/)

树干去重复，看的是nums[j] == nums[j-1]的 前一个数字和当前的数字是否一样，并且这个数字是已经被访问过了（false）
```python
class Solution(object):
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        ans = []
        path = [0] * n
        on_path = [False] * n
        def dfs(i:int) -> None:
            if i == n:
                ans.append(path.copy())
                return
            for j,on in enumerate(on_path):
                if j >= 1  and nums[j] == nums[j-1] and on_path[j-1] == False:
                    continue
                if not on:
                    path[i] = nums[j]
                    on_path[j] = True
                    dfs(i + 1)
                    on_path[j] = False
        dfs(0)
        return ans

```
# [53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/)


有很多方法，想到方法不重要，想到一开始的思路不重要，重要的是怎么去实现它


https://leetcode.cn/problems/maximum-subarray/solutions/2533977/qian-zhui-he-zuo-fa-ben-zhi-shi-mai-mai-abu71/

可以看看灵茶山艾府的题解


```python
class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        ans = -inf
        min_pre_sum = pre_sum = 0
        for x in nums:
            pre_sum += x  # 当前的前缀和
            ans = max(ans, pre_sum - min_pre_sum)  # 减去前缀和的最小值
            min_pre_sum = min(min_pre_sum, pre_sum)  # 维护前缀和的最小值
        return ans
```
# [54. 螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/)

```python
class Solution(object):
    def spiralOrder(self, matrix: List[List[int]]) -> List[int]:
        left,right,up,down = 0, len(matrix[0])-1,0,len(matrix)-1
        ans = []
        while True:
            for i in range(left,right+1):
                ans.append(matrix[up][i])
            up += 1
            if up > down:
                break

            for i in range(up, down + 1):
                ans.append(matrix[i][right])
            right -= 1
            if left > right:
                break

            for i in range(right, left-1, -1):
                ans.append(matrix[down][i])
            down-=1
            if up > down:
                break
            for i in range(down, up-1, -1):
                ans.append(matrix[i][left])
            left += 1
            if left > right:
                break
        return ans
```


# [310. 最小高度树](https://leetcode.cn/problems/minimum-height-trees/)

找到树的圆心，具体题解可以看英文版本，我觉得细致很多
https://leetcode.com/problems/minimum-height-trees/editorial/?envType=problem-list-v2&envId=rab78cw1

相连的只有一个节点就是leaf节点
```python
class Solution:
    def findMinHeightTrees(self, n: int, edges: List[List[int]]) -> List[int]:

        # edge cases
        if n <= 2:
            return [i for i in range(n)]

        # Build the graph with the adjacency list
        neighbors = [set() for i in range(n)]
        for start, end in edges:
            neighbors[start].add(end)
            neighbors[end].add(start)

        # Initialize the first layer of leaves
        leaves = []
        for i in range(n):
            if len(neighbors[i]) == 1:
                leaves.append(i)

        # Trim the leaves until reaching the centroids
        remaining_nodes = n
        while remaining_nodes > 2:
            remaining_nodes -= len(leaves)
            new_leaves = []
            # remove the current leaves along with the edges
            while leaves:
                leaf = leaves.pop()
                # the only neighbor left for the leaf node
                neighbor = neighbors[leaf].pop()
                # remove the only edge left
                neighbors[neighbor].remove(leaf)
                if len(neighbors[neighbor]) == 1:
                    new_leaves.append(neighbor)

            # prepare for the next round
            leaves = new_leaves

        # The remaining nodes are the centroids of the graph
        return leaves
```
## [207. 课程表](https://leetcode.cn/problems/course-schedule/)
拓扑排序经典题，找入度为0的点
```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 建图：pre -> course
        g = [[] for _ in range(numCourses)]
        indeg = [0] * numCourses
        for course, pre in prerequisites:
            g[pre].append(course)
            indeg[course] += 1

        # 初始入度为 0 的点入队
        q = deque(i for i, d in enumerate(indeg) if d == 0)

        visited = 0
        while q:
            u = q.popleft()
            visited += 1
            for v in g[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        return visited == numCourses
```
## [210. 课程表 II](https://leetcode.cn/problems/course-schedule-ii/)
一样，除了多了一个队列
```python
class Solution:
    def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
        graph = defaultdict(list)
        degree = [0] * numCourses
        queue = deque()
        result = []
        for course, pre in prerequisites:
            graph[pre].append(course)
            degree[course] += 1
        for index, value in enumerate(degree):
            if value == 0:
                queue.append(index)
        while len(queue) != 0:
            index = queue.popleft()
            for value in graph[index]:
                degree[value] -= 1
                if degree[value] == 0:
                    queue.append(value)
            result.append(index)
        if len(result) == numCourses:
            return result
        else:
            return []

```


# [56. 合并区间](https://leetcode.cn/problems/merge-intervals/)

有很多类似的题，看贪心章节，还有会议室，和会议室2也是常考题，见tt tag题单


主要是排序函数lambda表达式怎么写的问题
```python
    def merge(self, intervals: List[List[int]]) -> List[List[int]]:
        # 1️⃣ 先按照每个区间的起点排序
        intervals.sort(key=lambda x: x[0])

        merged = []
        for interval in intervals:
            # 如果结果列表为空，或当前区间和上一个不重叠，直接加入
            if not merged or merged[-1][1] < interval[0]:
                merged.append(interval)
            else:
                # 否则，有重叠，合并区间
                merged[-1][1] = max(merged[-1][1], interval[1])
        return merged
```


# [57. 插入区间](https://leetcode.cn/problems/insert-interval/)

两个区间 [a1, b1] 和 [a2, b2] 有重叠的条件是：
==a1 <= b2 and a2 <= b1==


```python
from typing import List

class Solution:
    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        res = []
        i = 0
        n = len(intervals)

        # 1️⃣ 把所有在 newInterval 左边、不重叠的区间加入结果
        while i < n and intervals[i][1] < newInterval[0]:
            res.append(intervals[i])
            i += 1

        # 2️⃣ 合并所有与 newInterval 重叠的区间
        while i < n and intervals[i][0] <= newInterval[1]:
            newInterval[0] = min(newInterval[0], intervals[i][0])
            newInterval[1] = max(newInterval[1], intervals[i][1])
            i += 1
        res.append(newInterval)

        # 3️⃣ 把剩下在右边的区间加入结果
        while i < n:
            res.append(intervals[i])
            i += 1

        return res
```



# [438. 找到字符串中所有字母异位词](https://leetcode.cn/problems/find-all-anagrams-in-a-string/)

不要太追求剪枝
```python
class Solution:
    def allZero(self, alpha: list[int]) -> bool:
        for num in alpha:
            if num != 0:
                return False
        return True
    def findAnagrams(self, s: str, p: str) -> List[int]:
        if len(p) > len(s):
            return []
        alpha = [0] * 26
        lenp = len(p)
        for ch in p:
            alpha[ord(ch) - ord("a")] += 1
        left, right = 0, lenp-1
        ans = []
        for i in range(left,right + 1):
            alpha[ord(s[i]) - ord("a")] -=1
        while right < len(s):
            if self.allZero(alpha):
                ans.append(left)
            alpha[ord(s[left]) - ord("a")] += 1
            left += 1
            right += 1
            if right >= len(s):
                break
            alpha[ord(s[right]) - ord("a")] -= 1
        return ans
```

# [62. 不同路径](https://leetcode.cn/problems/unique-paths/)
简单二维爬楼梯
```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[0] * n for _ in range(m)]
        for i in range(0, m):
            dp[i][0] = 1
        for i in range(0,n):
            dp[0][i] = 1
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[m-1][n-1]
```


# [322. 零钱兑换](https://leetcode.cn/problems/coin-change/)
背包问题统计次数，记得和求路径的组合总和区分一下，什么时候统计次数，什么时候统计路径
```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        dp = [float('inf')] * (amount + 1)  # 创建动态规划数组，初始值为正无穷大
        dp[0] = 0  # 初始化背包容量为0时的最小硬币数量为0

        for coin in coins:  # 遍历硬币列表，相当于遍历物品
            for i in range(coin, amount + 1):  # 遍历背包容量
                if dp[i - coin] != float('inf'):  # 如果dp[i - coin]不是初始值，则进行状态转移
                    dp[i] = min(dp[i - coin] + 1, dp[i])  # 更新最小硬币数量

        if dp[amount] == float('inf'):  # 如果最终背包容量的最小硬币数量仍为正无穷大，表示无解
            return -1
        return dp[amount]  # 返回背包容量为amount时的最小硬币数量

```


# [70. 爬楼梯](https://leetcode.cn/problems/climbing-stairs/)
east题，没什么好说的，组合总和4的基础题
```python
class Solution:
    def climbStairs(self, n: int) -> int:
        @cache
        def dfs(i:int)->int:
            if i < 0:
                return 0
            if i ==1:
                return 1
            if i == 2:
                return 2
            return dfs(i-2) + dfs(i-1)
        return  dfs(n)

    
        
```



# [67. 二进制求和](https://leetcode.cn/problems/add-binary/)

我的，写的不好，学一下加法怎么写


```python
class Solution:
    def addBinary(self, a: str, b: str) -> str:
        ans = []
        carry = 0
        i, j = len(a) - 1, len(b) - 1

        while i >= 0 or j >= 0:
            total = carry
            if i >= 0:
                total += int(a[i])
                i -= 1
            if j >= 0:
                total += int(b[j])
                j -= 1

            ans.append(str(total % 2))
            carry = total // 2

        if carry == 1:
            ans.append("1")

        return "".join(reversed(ans))
```

# [199. 二叉树的右视图](https://leetcode.cn/problems/binary-tree-right-side-view/)

可以层序遍历，可以dfs.
0x3f的题解是dfs，先遍历右子树，再遍历左子树，这样就是右优先布局
```python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if root == None:
            return []
        q = deque()
        q.append(root)
        ans = []
        while q:
            lenLayer = len(q)
            for i in range(lenLayer):
                if i == lenLayer - 1:
                    ans.append(q[0].val)
                node = q.popleft()
                if node.left:
                    q.append(node.left)
                if node.right:
                    q.append(node.right)
        return ans
```



# [200. 岛屿数量](https://leetcode.cn/problems/number-of-islands/)

x 和y把我有点绕进去了，其实没必要哪个是x哪个是y
```python
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        dirct = [[-1, 0], [1, 0], [0, 1], [0, - 1]]
        ans = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    ans+=1
                    grid[i][j] = "2"
                    q  = deque([[i,j]])
                    while q:
                        pos = q.popleft()
                        for nx, ny in dirct:
                            x = pos[0] + nx
                            y = pos[1] + ny
                            if x >= 0 and x < len(grid) and y >=0 and y < len(grid[0]) and grid[x][y] == "1":
                                grid[x][y] = "2"
                                q.append([x,y])
        return ans
```


# [1235. 规划兼职工作](https://leetcode.cn/problems/maximum-profit-in-job-scheduling/)

我的解法，爆内存了。注意按照endtime排序，因为是从最后一个数字开始看，然后不断更新endtime。爆内存是因为我存了end_time，可以优化

其实我们并不需要end_time，我们可以根据end_time，直接找到满足条件的下一个下标，这样就可以直接进行搜索了
```python
class Solution:
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        time_intervel = []
        for i in range(len(startTime)):
            time_intervel.append([startTime[i],endTime[i],profit[i]])
        time_intervel.sort(key=lambda x: x[1])
        @cache
        def dfs(end_time: int, i: int):
            if i < 0 or end_time == 1:
                return 0
            select = 0
            if time_intervel[i][1] <= end_time:
                select = dfs(time_intervel[i][0], i-1) + time_intervel[i][2]
            notSelect = dfs(end_time, i-1)
            return max(select,notSelect)
        return dfs(time_intervel[-1][1],len(time_intervel)-1)
```


```python
from bisect import bisect_right
from typing import List

class Solution:
    def jobScheduling(self, startTime: List[int], endTime: List[int], profit: List[int]) -> int:
        # 按结束时间排序
        jobs = sorted(zip(endTime, startTime, profit))
        ends = [e for e, _, _ in jobs]  # 单独提取结束时间序列

        f = [0] * (len(jobs) + 1)#注意这里的dp是jobs的个数，而不是第i天
        for i, (end, st, p) in enumerate(jobs):
            # 找到所有 end <= st 的最后一个位置（即满足条件的任务个数）， hi=i 表示二分上界为 i（默认为 n）,开区间，不会取到i
            j = bisect_right(ends, st, hi=i)
            # j 表示可以承接的任务数量
            f[i + 1] = max(f[i], f[j] + p)
        return f[-1]
```

## [1751. 最多可以参加的会议数目 II](https://leetcode.cn/problems/maximum-number-of-events-that-can-be-attended-ii/)
见注释，有点类似于买卖股票+上面的结合体。注意这里的dp是jobs的数量，而不是天数量

```python
from bisect import bisect_right
from typing import List

class Solution:
    def maxValue(self, events: List[List[int]], k: int) -> int:
        # 1️⃣ 按“结束时间”排序，确保前面的会议都在后面的会议之前结束。
        #    这样我们就可以用二分查找快速找到“当前会议之前最后一个不冲突的会议”。
        events.sort(key=lambda x: x[1])

        # 2️⃣ 提前抽取所有会议的结束时间，用于后续二分查找。
        ends = [e[1] for e in events]

        # 3️⃣ 计算 prev[i]：表示在第 i 个会议开始之前，最后一个不冲突的会议下标（如果没有则为 -1）。
        #    bisect_right(ends, s-1) 会返回“第一个 end > s-1”的位置，减 1 就是最后一个 end <= s-1 的会议。
        #    ⚠️ 这是关键点——确保会议间没有重叠（包括同一天）。
        prev = [bisect_right(ends, s - 1) - 1 for s, _, _ in events]

        n = len(events)
        # 4️⃣ dp[i][t] 表示：只考虑前 i 个会议，最多参加 t 个会议，能获得的最大价值。
        #    dp[0][*] = 0 表示没选会议；dp[*][0] = 0 表示不允许选会议。
        dp = [[0] * (k + 1) for _ in range(n + 1)]

        # 5️⃣ 枚举会议 i（1-based，因为 dp[0] 表示空集）
        for i in range(1, n + 1):
            s, e, v = events[i - 1]  # 当前会议的开始时间、结束时间、价值
            for t in range(1, k + 1):
                # ① 不选当前会议：继承前一个状态
                not_take = dp[i - 1][t]

                # ② 选当前会议：加上当前会议价值 + 前一个不冲突会议的最优值
                #    prev[i-1] 是基于 0-based 下标，因此要 +1 对齐到 dp 的行。
                take = dp[prev[i - 1] + 1][t - 1] + v

                # ③ 取两者最大
                dp[i][t] = max(not_take, take)

        # 6️⃣ 返回前 n 个会议、最多选 k 个的最优结果。
        return dp[n][k]
```
## [2008. 出租车的最大盈利](https://leetcode.cn/problems/maximum-earnings-from-taxi/)
一样的题目了
```python
class Solution:
    def maxTaxiEarnings(self, n: int, rides: List[List[int]]) -> int:
        rides.sort(key= lambda  x:(x[1],x[2]))
        dp = [0] * (len(rides)+1)
        ends = [e for _,e,_ in rides]
        for i in range(1, len(rides)+1):
            st, et, val = rides[i-1][0], rides[i-1][1], rides[i-1][2]
            idx = bisect_right(ends,st,hi=i-1)
            dp[i] = max(
                dp[i-1],
                dp[idx]+et-st+val
            )
        return dp[-1]
```
# [75. 颜色分类](https://leetcode.cn/problems/sort-colors)

看看荷兰国旗，快排怎么写，感觉和快排不是一个东西


本题实际上是 「荷兰国旗问题」，它模拟的是荷兰国旗的三种颜色排列问题。假设有一个包含三种颜色的数组，例如：
	•	红 red → 0
	•	白 white → 1
	•	蓝 blue → 2

目标就是原地将这些颜色按红、白、蓝的顺序排序。

Dijkstra 提出的经典解法，利用了三个指针，它们的定义为：
	•	low：0 区的下一个位置（用于放置 0）
	•	mid：当前正在处理的位置
	•	high：2 区的前一个位置（用于放置 2）

它们分别初始化为 0, 0, n - 1。

处理逻辑：
	•	当 arr[mid] == 0：把它与 arr[low] 交换，low += 1，mid += 1
	•	当 arr[mid] == 1：直接跳过，mid += 1
	•	当 arr[mid] == 2：把它与 arr[high] 交换，high -= 1（注意 mid 不动，因为交换过来的值可能还需要判断）

---

本质： 三个指针用于划分数组，左侧 [0, low) 是 0 区域，中间 [low, mid) 是 1 区域，剩下 [high, n) 的是 2 区域。

流程： 指针 mid 作为活动指针，而 low、high 则是作为标记位。当前 low 左侧和 high 右侧都已经处理完毕，中间部分等待处理。如果 mid > high，说明所有元素处理完成。

适用范围： 类别数量固定且较少的「划分」问题。

```python
class Solution:
    def sortColors(self, nums: List[int]) -> None:
        # 分别指向0，1，2区域
        low, mid, high = 0, 0, len(nums) - 1
        # mid是活动指针
        while mid <= high:
            if nums[mid] == 0:  # 说明要交换到0区
                nums[low], nums[mid] = nums[mid], nums[low]
                low += 1
                mid += 1
            elif nums[mid] == 1:  # 正好
                mid += 1
            else:  # 否则交换到2区
                nums[mid], nums[high] = nums[high], nums[mid]
                high -= 1
```
## [912. 排序数组](https://leetcode.cn/problems/sort-an-array/)
[[快排]]
主要思想就是找一个数，比他小的都在这个数左边，比他大的都在这个数右边
每次递归搞定了一个数的顺序问题

```text
在arr[L..R]范围上，进行快速排序的过程：

1）用arr[R]对该范围做partition，<= arr[R]的数在左部分并且保证arr[R]最后来到左部分
    的最后一个位置，记为M； > arr[R]的数在右部分（arr[M+1..R]）
2）对arr[L..M-1]进行快速排序(递归)
3）对arr[M+1..R]进行快速排序(递归)

因为每一次partition都会搞定一个数的位置且不会再变动，所以排序能完成
```


https://leetcode.cn/problems/sort-an-array/
在写快排的时候，一定要注意灵神题解的一些细节，随机，全部等的时候的细节，防止特殊情况的退化超时

```python
class Solution:
    def partition(self, nums: List[int], left: int, right: int) -> int:
        """
        在子数组 [left, right] 中随机选择一个基准元素 pivot
        根据 pivot 重新排列子数组 [left, right]
        重新排列后，<= pivot 的元素都在 pivot 的左侧，>= pivot 的元素都在 pivot 的右侧
        返回 pivot 在重新排列后的 nums 中的下标
        特别地，如果子数组的所有元素都等于 pivot，我们会返回子数组的中心下标，避免退化
        """

        # 1. 在子数组 [left, right] 中随机选择一个基准元素 pivot
        i = randint(left, right)
        pivot = nums[i]
        # 把 pivot 与子数组第一个元素交换，避免 pivot 干扰后续划分，从而简化实现逻辑
        nums[i], nums[left] = nums[left], nums[i]

        # 2. 相向双指针遍历子数组 [left + 1, right]
        # 循环不变量：在循环过程中，子数组的数据分布始终如下图
        # [ pivot | <=pivot | 尚未遍历 | >=pivot ]
        #   ^                 ^     ^         ^
        #   left              i     j         right

        i, j = left + 1, right
        while True:
            while i <= j and nums[i] < pivot:
                i += 1
            # 此时 nums[i] >= pivot

            while i <= j and nums[j] > pivot:
                j -= 1
            # 此时 nums[j] <= pivot

            if i >= j:
                break

            # 维持循环不变量
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1

        # 循环结束后
        # [ pivot | <=pivot | >=pivot ]
        #   ^             ^   ^     ^
        #   left          j   i     right

        # 3. 把 pivot 与 nums[j] 交换，完成划分（partition）
        # 为什么与 j 交换？
        # 如果与 i 交换，可能会出现 i = right + 1 的情况，已经下标越界了，无法交换
        # 另一个原因是如果 nums[i] > pivot，交换会导致一个大于 pivot 的数出现在子数组最左边，不是有效划分
        # 与 j 交换，即使 j = left，交换也不会出错
        nums[left], nums[j] = nums[j], nums[left]

        # 交换后
        # [ <=pivot | pivot | >=pivot ]
        #               ^
        #               j

        # 返回 pivot 的下标
        return j

    def sortArray(self, nums: List[int]) -> List[int]:
        # 快速排序子数组 [left, right]
        def quick_sort(left: int, right: int) -> None:
            # 优化：如果子数组已是升序，直接返回
            # 也可以写 if all(nums[i] <= nums[i + 1] for i in range(left, right)): return
            ordered = True
            for i in range(left, right):
                if nums[i] > nums[i + 1]:
                    ordered = False
                    break
            if ordered:
                return

            i = self.partition(nums, left, right)  # 划分子数组
            quick_sort(left, i - 1)   # 排序在 pivot 左侧的元素
            quick_sort(i + 1, right)  # 排序在 pivot 右侧的元素

        quick_sort(0, len(nums) - 1)
        return nums
```

## [215. 数组中的第K个最大元素](https://leetcode.cn/problems/kth-largest-element-in-an-array/)

```python
class Solution:
    def partition(self, nums: List[int], left: int, right: int) -> int:
        """
        在子数组 [left, right] 中随机选择一个基准元素 pivot
        根据 pivot 重新排列子数组 [left, right]
        重新排列后，<= pivot 的元素都在 pivot 的左侧，>= pivot 的元素都在 pivot 的右侧
        返回 pivot 在重新排列后的 nums 中的下标
        特别地，如果子数组的所有元素都等于 pivot，我们会返回子数组的中心下标，避免退化
        """

        # 1. 在子数组 [left, right] 中随机选择一个基准元素 pivot
        i = randint(left, right)
        pivot = nums[i]
        # 把 pivot 与子数组第一个元素交换，避免 pivot 干扰后续划分，从而简化实现逻辑
        nums[i], nums[left] = nums[left], nums[i]

        # 2. 相向双指针遍历子数组 [left + 1, right]
        # 循环不变量：在循环过程中，子数组的数据分布始终如下图
        # [ pivot | <=pivot | 尚未遍历 | >=pivot ]
        #   ^                 ^     ^         ^
        #   left              i     j         right

        i, j = left + 1, right
        while True:
            while i <= j and nums[i] < pivot:
                i += 1
            # 此时 nums[i] >= pivot

            while i <= j and nums[j] > pivot:
                j -= 1
            # 此时 nums[j] <= pivot

            if i >= j:
                break

            # 维持循环不变量
            nums[i], nums[j] = nums[j], nums[i]
            i += 1
            j -= 1

        # 循环结束后
        # [ pivot | <=pivot | >=pivot ]
        #   ^             ^   ^     ^
        #   left          j   i     right

        # 3. 把 pivot 与 nums[j] 交换，完成划分（partition）
        # 为什么与 j 交换？
        # 如果与 i 交换，可能会出现 i = right + 1 的情况，已经下标越界了，无法交换
        # 另一个原因是如果 nums[i] > pivot，交换会导致一个大于 pivot 的数出现在子数组最左边，不是有效划分
        # 与 j 交换，即使 j = left，交换也不会出错
        nums[left], nums[j] = nums[j], nums[left]

        # 交换后
        # [ <=pivot | pivot | >=pivot ]
        #               ^
        #               j

        # 返回 pivot 的下标
        return j

    def findKthLargest(self, nums: list[int], k: int) -> int:
        n = len(nums)
        target_index = n - k  # 第 k 大元素在升序数组中的下标是 n - k
        left, right = 0, n - 1  # 闭区间
        while True:
            i = self.partition(nums, left, right)
            if i == target_index:
                # 找到第 k 大元素
                return nums[i]
            if i > target_index:
                # 第 k 大元素在 [left, i - 1] 中
                right = i - 1
            else:
                # 第 k 大元素在 [i + 1, right] 中
                left = i + 1
```
## 荷兰国旗问题
[[荷兰国旗问题]]
给定一个数组arr，和一个整数num。请把小于num的数放在数组的左边，等于num的数放在中间，大于num的数放在数组的右边。

要求额外空间复杂度O(1)，时间复杂度O(N) 

重点理解上面是怎么分区的，如果懂了上面的问题如何分区，就可以做了

上题是把区分成了:
`<=pivot|pivot|>=pivot`

荷兰问题是把问题分成了
`< pivot | pivot | > pivot`

看这个题解吧，我是蠢货
https://leetcode.cn/problems/sort-colors/solutions/3679173/tu-jie-he-lan-guo-qi-san-zhi-zhen-wen-ti-k690/

# [78. 子集](https://leetcode.cn/problems/subsets/)

发疯了，想着循环
```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        ans = []
        path = []

        def dfs(i: int) -> None:
            if i == n:  # 子集构造完毕
                ans.append(path.copy())  # 复制 path，也可以写 path[:]
                return
                
            # 不选 nums[i]
            dfs(i + 1)
            
            # 选 nums[i]
            path.append(nums[i])
            dfs(i + 1)
            path.pop()  # 恢复现场

        dfs(0)
        return ans
```

##  [77. 组合](https://leetcode.cn/problems/combinations/)

这个也不用for
```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        self.ans = []
        self.path = []
        def dfs(i:int):
            if len(self.path) == k:
                self.ans.append(self.path.copy())
                return
            if i > n:
                return
            self.path.append(i)
            dfs(i+1)
            self.path.pop()
            dfs(i+1)
        dfs(1)
        return self.ans
```

## [46. 全排列](https://leetcode.cn/problems/permutations/)


[[#[46. 全排列](https //leetcode.cn/problems/permutations/)]]



# [79. 单词搜索](https://leetcode.cn/problems/word-search/)

简单的图搜索
注意模版的单词搜索，还有判断是否visited过

```python
class Solution:
    def exist(self, board: List[List[str]], word: str) -> bool:
        m = len(board)
        n = len(board[0])
        self.direct = [[-1,0],[1,0],[0,-1],[0,1]]
        self.visited = [[False] * n for _ in range(m)]
        def dfs(i:int, j:int, k:int) -> bool:
            if k >= len(word):
                return True
            for x, y in self.direct:
                nx = i + x
                ny = j + y
                if 0 <= nx < len(board) and 0 <= ny < len(board[0]) and not self.visited[nx][ny]:
                    if board[nx][ny] == word[k]:
                        self.visited[nx][ny] = True
                        if dfs(nx,ny,k+1):
                            return True
                        self.visited[nx][ny] = False
            return False
        for i in range(m):
            for j in range(n):
                if board[i][j] == word[0]:
                    self.visited[i][j] = True
                    if dfs(i,j,1):
                        return True
                    self.visited[i][j] = False

        return False
```


# [206. 反转链表](https://leetcode.cn/problems/reverse-linked-list/)
注意细节，每次循环结束，最终prev会指向反转后的头部。 head 和 next 会指向同一个位置，是现在prev原本的next
```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        next = None
        while head != None:
            next = head.next
            head.next = prev
            prev = head
            head = next
        return prev
```

# [207. 课程表](https://leetcode.cn/problems/course-schedule/)

拓扑排序
```python
class Solution:
    def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
        # 建图：pre -> course
        g = [[] for _ in range(numCourses)]
        indeg = [0] * numCourses
        for course, pre in prerequisites:
            g[pre].append(course)
            indeg[course] += 1

        # 初始入度为 0 的点入队
        q = deque(i for i, d in enumerate(indeg) if d == 0)

        visited = 0
        while q:
            u = q.popleft()
            visited += 1
            for v in g[u]:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

        return visited == numCourses
```

# [208. 实现 Trie (前缀树)](https://leetcode.cn/problems/implement-trie-prefix-tree/)
很简单的东西，26叉树，一个叉是一个节点，一直叉开作为前缀就好了

```python
class Node:
    __slots__ = 'son', 'end'

    def __init__(self):
        self.son = {}
        self.end = False

class Trie:
    def __init__(self):
        self.root = Node()

    def insert(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.son:  # 无路可走？
                cur.son[c] = Node()  # 那就造路！
            cur = cur.son[c]
        cur.end = True

    def find(self, word: str) -> int:
        cur = self.root
        for c in word:
            if c not in cur.son:  # 道不同，不相为谋
                return 0
            cur = cur.son[c]
        # 走过同样的路（2=完全匹配，1=前缀匹配）
        return 2 if cur.end else 1

    def search(self, word: str) -> bool:
        return self.find(word) == 2

    def startsWith(self, prefix: str) -> bool:
        return self.find(prefix) != 0
```

# [721. 账户合并](https://leetcode.cn/problems/accounts-merge/)
并查集不太会，先看下面的题目学习基础的思路
看图吧，上面有说明白许多并查集的问题
[[图]]
# [84. 柱状图中最大的矩形](https://leetcode.cn/problems/largest-rectangle-in-histogram/)

想不出收汁的办法
思想是每一条柱子都找他的最左边和最右边，可以用单调栈找下一个比自身小的元素实现
```python

class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = []
        area = 0
        for idx, val in enumerate(heights):
            while stack and heights[stack[-1]] > val:
                right = idx
                heightIdx = stack.pop()
                if not stack:
                    left = -1
                else:
                    left = stack[-1]
                area = max(area, heights[heightIdx] * (right-left-1))
            stack.append(idx)
        right = len(heights)
        while stack:
            heightIdx = stack.pop()
            if not stack:
                left = -1
            else:
                left = stack[-1]
            area = max(area, heights[heightIdx] * (right - left - 1))
        return area
```

# [217. 存在重复元素](https://leetcode.cn/problems/contains-duplicate/)
一个set解决的事
```python
class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        st = set()
        for x in nums:
            if x in st:
                return True
            st.add(x)
        return False
```



# [733. 图像渲染](https://leetcode.cn/problems/flood-fill/)

和岛屿问题一样

```python
class Solution:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
        if image[sr][sc] != newColor:
            old, image[sr][sc] = image[sr][sc], newColor
            for i, j in zip((sr, sr+1, sr, sr-1), (sc+1, sc, sc-1, sc)):
                if 0 <= i < len(image) and 0 <= j < len(image[0]) and image[i][j] == old:
                    self.floodFill(image, i, j, newColor)
        return image
```



# [224. 基本计算器](https://leetcode.cn/problems/basic-calculator/)

我们可以使用两个栈 nums 和 ops 。

nums ： 存放所有的数字
ops ：存放所有的数字以外的操作，+/- 也看做是一种操作
	然后从前往后做，对遍历到的字符做分情况讨论：
	空格 : 跳过
	( : 直接加入 ops 中，等待与之匹配的 )
	) : 使用现有的 nums 和 ops 进行计算，直到遇到左边最近的一个左括号为止，计算结果放到 nums
	数字 : 从当前位置开始继续往后取，将整一个连续数字整体取出，加入 nums
	+/- : 需要将操作放入 ops 中。在放入之前先把栈内可以算的都算掉，使用现有的 nums 和 ops 进行计算，直到没有操作或者遇到左括号，计算结果放到 nums

一些细节：
由于第一个数可能是负数，为了减少边界判断。一个小技巧是先往 nums 添加一个 0
为防止 () 内出现的首个字符为运算符，将所有的空格去掉，并将 (- 替换为 (0-，(+ 替换为 (0+（当然也可以不进行这样的预处理，将这个处理逻辑放到循环里去做）

```java
    class Solution {
        public int calculate(String s) {
            // 存放所有的数字
            Deque<Integer> nums = new ArrayDeque<>();
            // 为了防止第一个数为负数，先往 nums 加个 0
            nums.addLast(0);
            // 将所有的空格去掉
            s = s.replaceAll(" ", "");
            // 存放所有的操作，包括 +/-
            Deque<Character> ops = new ArrayDeque<>();
            int n = s.length();
            char[] cs = s.toCharArray();
            for (int i = 0; i < n; i++) {
                char c = cs[i];
                if (c == '(') {
                    ops.addLast(c);
                } else if (c == ')') {
                    // 计算到最近一个左括号为止
                    while (!ops.isEmpty()) {
                        char op = ops.peekLast();
                        if (op != '(') {
                            calc(nums, ops);
                        } else {
                            ops.pollLast();
                            break;
                        }
                    }
                } else {
                    if (isNum(c)) {
                        int u = 0;
                        int j = i;
                        // 将从 i 位置开始后面的连续数字整体取出，加入 nums
                        while (j < n && isNum(cs[j])) {
                            u = u * 10 + (int)(cs[j++] - '0');
                        }
                        nums.addLast(u);
                        i = j - 1;
                    } else {
                        if (i > 0 && (cs[i - 1] == '(' || cs[i - 1] == '+' || cs[i - 1] == '-')) {
                            nums.addLast(0);
                        }
                        // 有一个新操作要入栈时，先把栈内可以算的都算了
                        while (!ops.isEmpty() && ops.peekLast() != '(') {
                            calc(nums, ops);
                        }
                        ops.addLast(c);
                    }
                }
            }
            while (!ops.isEmpty()) {calc(nums, ops);}
            return nums.peekLast();
        }
        void calc(Deque<Integer> nums, Deque<Character> ops) {
            if (nums.isEmpty() || nums.size() < 2) {return;}
            if (ops.isEmpty()) {return;}
            int b = nums.pollLast(), a = nums.pollLast();
            char op = ops.pollLast();
            nums.addLast(op == '+' ? a + b : a - b);
        }
        boolean isNum(char c) {
            return Character.isDigit(c);
        }
    }
    
```

```python
from collections import deque

class Solution:
    def calculate(self, s: str) -> int:
        # 存放所有的数字
        nums = deque()
        # 防止第一个数为负数，先加 0
        nums.append(0)

        # 去掉所有空格
        s = s.replace(" ", "")
        n = len(s)
        cs = s

        # 存放所有操作符 + / -
        ops = deque()

        i = 0
        while i < n:
            c = cs[i]

            if c == '(':
                ops.append(c)

            elif c == ')':
                # 计算到最近一个 '(' 为止
                while ops:
                    if ops[-1] != '(':
                        self.calc(nums, ops)
                    else:
                        ops.pop()
                        break

            elif c.isdigit():
                u = 0
                j = i
                # 读取连续数字
                while j < n and cs[j].isdigit():
                    u = u * 10 + (ord(cs[j]) - ord('0'))
                    j += 1
                nums.append(u)
                i = j - 1

            else:
                # 处理一元 +/-（如 -1, (-2)）
                if i > 0 and (cs[i - 1] == '(' or cs[i - 1] == '+' or cs[i - 1] == '-'):
                    nums.append(0)

                # 当前操作符入栈前，先把能算的算了
                while ops and ops[-1] != '(':
                    self.calc(nums, ops)
                ops.append(c)

            i += 1

        # 处理剩余操作
        while ops:
            self.calc(nums, ops)

        return nums[-1]

    def calc(self, nums: deque, ops: deque) -> None:
        if len(nums) < 2 or not ops:
            return
        b = nums.pop()
        a = nums.pop()
        op = ops.pop()
        nums.append(a + b if op == '+' else a - b)
```
## [227. 基本计算器 II](https://leetcode.cn/problems/basic-calculator-ii/)

继续看三叶的
```python
from collections import deque
from typing import Deque


class Solution:
    def __init__(self):
        # 运算符优先级
        self.priority = {
            '-': 1,
            '+': 1,
            '*': 2,
            '/': 2,
            '%': 2,
            '^': 3
        }

    def calculate(self, s: str) -> int:
        # 去掉所有空格
        s = s.replace(" ", "")
        n = len(s)

        # 数字栈
        nums: Deque[int] = deque()
        # 防止第一个数是负数
        nums.append(0)

        # 运算符栈
        ops: Deque[str] = deque()

        i = 0
        while i < n:
            c = s[i]

            if c == '(':
                ops.append(c)

            elif c == ')':
                # 计算到最近一个 '('
                while ops:
                    if ops[-1] != '(':
                        self.calc(nums, ops)
                    else:
                        ops.pop()
                        break

            elif c.isdigit():
                u = 0
                j = i
                # 读取连续数字
                while j < n and s[j].isdigit():
                    u = u * 10 + (ord(s[j]) - ord('0'))
                    j += 1
                nums.append(u)
                i = j - 1

            else:
                # 处理一元 +/-，如 (-2)、(+3)
                if i > 0 and s[i - 1] in ('(', '+', '-'):
                    nums.append(0)

                # 当前运算符入栈前，先把能算的算了
                while ops and ops[-1] != '(':
                    prev = ops[-1]
                    if self.priority[prev] >= self.priority[c]:
                        self.calc(nums, ops)
                    else:
                        break

                ops.append(c)

            i += 1

        # 清空剩余运算
        while ops:
            self.calc(nums, ops)

        return nums[-1]

    def calc(self, nums: Deque[int], ops: Deque[str]) -> None:
        if len(nums) < 2 or not ops:
            return

        b = nums.pop()
        a = nums.pop()
        op = ops.pop()

        if op == '+':
            ans = a + b
        elif op == '-':
            ans = a - b
        elif op == '*':
            ans = a * b
        elif op == '/':
            # Python 要向 0 截断
            ans = int(a / b)
        elif op == '^':
            ans = pow(a, b)
        elif op == '%':
            ans = a % b
        else:
            raise ValueError(f"Unknown operator: {op}")

        nums.append(ans)
```
## [772. Basic Calculator III](https://leetcode.com/problems/basic-calculator-iii/)
和2一样的思路

```python
from collections import deque
from typing import Deque


class Solution:
    def __init__(self):
        # 运算符优先级
        self.priority = {
            '-': 1,
            '+': 1,
            '*': 2,
            '/': 2,
            '%': 2,
            '^': 3
        }

    def calculate(self, s: str) -> int:
        # 去掉所有空格
        s = s.replace(" ", "")
        n = len(s)

        # 数字栈
        nums: Deque[int] = deque()
        # 防止第一个数是负数
        nums.append(0)

        # 运算符栈
        ops: Deque[str] = deque()

        i = 0
        while i < n:
            c = s[i]

            if c == '(':
                ops.append(c)

            elif c == ')':
                # 计算到最近一个 '('
                while ops:
                    if ops[-1] != '(':
                        self.calc(nums, ops)
                    else:
                        ops.pop()
                        break

            elif c.isdigit():
                u = 0
                j = i
                # 读取连续数字
                while j < n and s[j].isdigit():
                    u = u * 10 + (ord(s[j]) - ord('0'))
                    j += 1
                nums.append(u)
                i = j - 1

            else:
                # 处理一元 +/-，如 (-2)、(+3)
                if i > 0 and s[i - 1] in ('(', '+', '-'):
                    nums.append(0)

                # 当前运算符入栈前，先把能算的算了
                while ops and ops[-1] != '(':
                    prev = ops[-1]
                    if self.priority[prev] >= self.priority[c]:
                        self.calc(nums, ops)
                    else:
                        break

                ops.append(c)

            i += 1

        # 清空剩余运算
        while ops:
            self.calc(nums, ops)

        return nums[-1]

    def calc(self, nums: Deque[int], ops: Deque[str]) -> None:
        if len(nums) < 2 or not ops:
            return

        b = nums.pop()
        a = nums.pop()
        op = ops.pop()

        if op == '+':
            ans = a + b
        elif op == '-':
            ans = a - b
        elif op == '*':
            ans = a * b
        elif op == '/':
            # Python 要向 0 截断
            ans = int(a / b)
        elif op == '^':
            ans = pow(a, b)
        elif op == '%':
            ans = a % b
        else:
            raise ValueError(f"Unknown operator: {op}")

        nums.append(ans)
```
## [770. 基本计算器 IV](https://leetcode.cn/problems/basic-calculator-iv/)
太难了，考了就g不做
```python
class Poly(collections.Counter):
    def __add__(self, other):
        self.update(other)
        return self

    def __sub__(self, other):
        self.update({k: -v for k, v in other.items()})
        return self

    def __mul__(self, other):
        ans = Poly()
        for k1, v1 in self.items():
            for k2, v2 in other.items():
                ans.update({tuple(sorted(k1 + k2)): v1 * v2})
        return ans

    def evaluate(self, evalmap):
        ans = Poly()
        for k, c in self.items():
            free = []
            for token in k:
                if token in evalmap:
                    c *= evalmap[token]
                else:
                    free.append(token)
            ans[tuple(free)] += c
        return ans

    def to_list(self):
        return ["*".join((str(v),) + k)
                for k, v in sorted(self.items(),
                    key = lambda (k, v): (-len(k), k, v))
                if v]

class Solution(object):
    def basicCalculatorIV(self, expression, evalvars, evalints):
        evalmap = dict(zip(evalvars, evalints))

        def combine(left, right, symbol):
            if symbol == '+': return left + right
            if symbol == '-': return left - right
            if symbol == '*': return left * right
            raise

        def make(expr):
            ans = Poly()
            if expr.isdigit():
                ans.update({(): int(expr)})
            else:
                ans[(expr,)] += 1
            return ans

        def parse(expr):
            bucket = []
            symbols = []
            i = 0
            while i < len(expr):
                if expr[i] == '(':
                    bal = 0
                    for j in xrange(i, len(expr)):
                        if expr[j] == '(': bal += 1
                        if expr[j] == ')': bal -= 1
                        if bal == 0: break
                    bucket.append(parse(expr[i+1:j]))
                    i = j
                elif expr[i].isalnum():
                    for j in xrange(i, len(expr)):
                        if expr[j] == ' ':
                            bucket.append(make(expr[i:j]))
                            break
                    else:
                        bucket.append(make(expr[i:]))
                    i = j
                elif expr[i] in '+-*':
                    symbols.append(expr[i])
                i += 1

            for i in xrange(len(symbols) - 1, -1, -1):
                if symbols[i] == '*':
                    bucket[i] = combine(bucket[i], bucket.pop(i+1),
                                        symbols.pop(i))

            if not bucket: return Poly()
            ans = bucket[0]
            for i, symbol in enumerate(symbols, 1):
                ans = combine(ans, bucket[i], symbol)

            return ans

        P = parse(expression).evaluate(evalmap)
        return P.to_list()
```
# [98. 验证二叉搜索树](https://leetcode.cn/problems/validate-binary-search-tree/)


写的不够清晰，gpt写得很清晰
```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        self.prev = None

        def dfs(node: Optional[TreeNode]) -> bool:
            if not node:
                return True
            # 1️⃣ 递归遍历左子树
            if not dfs(node.left):
                return False

            # 2️⃣ 检查当前节点是否比前一个节点大
            if self.prev and node.val <= self.prev.val:
                return False

            # 3️⃣ 更新 prev 为当前节点
            self.prev = node

            # 4️⃣ 继续遍历右子树
            return dfs(node.right)

        return dfs(root)
```


# [226. 翻转二叉树](https://leetcode.cn/problems/invert-binary-tree/)

看懂逻辑不难

```python
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        def dfs(root: Optional[TreeNode]) -> Optional[TreeNode]:
            if root == None:
                return None
            left = dfs(root.left)
            right = dfs(root.right)
            root.right = left
            root.left= right
            return root
        return dfs(root)
```


# [102. 二叉树的层序遍历](https://leetcode.cn/problems/binary-tree-level-order-traversal/)
写的时候神智不清了
bfs

```python
from collections import deque

class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []

        queue = deque([root])
        ans = []

        while queue:
            temp = []
            for _ in range(len(queue)):
                node = queue.popleft()          # ✅ 取出当前节点
                temp.append(node.val)           # 加入当前层结果
                if node.left:                   # ✅ 访问 node.left 而不是 root.left
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
            ans.append(temp)

        return ans
```

# [230. 二叉搜索树中第 K 小的元素](https://leetcode.cn/problems/kth-smallest-element-in-a-bst/)
换个方法便利
```python
class Solution:
    def kthSmallest(self, root: Optional[TreeNode], k: int) -> int:
        self.k = k
        def dfs(root: Optional[TreeNode]) -> int:
            if root == None:
                return -1
            left = dfs(root.left)
            if left != -1:
                return left
            self.k -=1
            if self.k == 0:
                return root.val
            return dfs(root.right)
        return dfs(root)

```


# [104. 二叉树的最大深度](https://leetcode.cn/problems/maximum-depth-of-binary-tree/)

dfs标准题


```python
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if root == None:
            return 0
        return 1 + max(self.maxDepth(root.left),self.maxDepth(root.right))
```




# [232. 用栈实现队列](https://leetcode.cn/problems/implement-queue-using-stacks/)
栈A倒出来之后，当B是空的才需要再倒，不然没必要再倒出来。因为B已经调整好顺序了，A只要继续按照顺序添加就好了
```python
class MyQueue:

    def __init__(self):
        self.A, self.B = [], []

    def push(self, x: int) -> None:
        self.A.append(x)

    def pop(self) -> int:
        peek = self.peek()
        self.B.pop()
        return peek

    def peek(self) -> int:
        if self.B: return self.B[-1]
        if not self.A: return -1
        # 将栈 A 的元素依次移动至栈 B
        while self.A:
            self.B.append(self.A.pop())
        return self.B[-1]

    def empty(self) -> bool:
        return not self.A and not self.B

```


# [105. 从前序与中序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

前序的第一个元素总是树的root，这个root 对应中序遍历的数组里就是左半子树和右半子树的切割点，依次递归就好了。

```python
class Solution:
    def buildTree(self, preorder: List[int], inorder: List[int]) -> Optional[TreeNode]:
        index = {x: i for i, x in enumerate(inorder)}

        def dfs(pre_l: int, pre_r: int, in_l: int, in_r: int) -> Optional[TreeNode]:
            if pre_l == pre_r:  # 空节点
                return None
            left_size = index[preorder[pre_l]] - in_l  # 左子树的大小
            left = dfs(pre_l + 1, pre_l + 1 + left_size, in_l, in_l + left_size)
            right = dfs(pre_l + 1 + left_size, pre_r, in_l + 1 + left_size, in_r)
            return TreeNode(preorder[pre_l], left, right)

        return dfs(0, len(preorder), 0, len(inorder))  # 左闭右开区间
```


## [106. 从中序与后序遍历序列构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-inorder-and-postorder-traversal/)
一样的思路，这次改成最后一个元素是root
我写的左闭 右闭的算法，感觉没写好边界条件，主要还是构建RootNode这一步的问题
```python
class Solution:
    def buildTree(self, inorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        index = {x: i for i, x in enumerate(inorder)}
        # [9 15 7 20 3]
        # [9 3 15 20 7]
        def dfs(post_l: int, post_r: int, in_l: int, in_r: int) -> Optional[TreeNode]:
            if post_l > post_r:
                return None
            root_index = index[postorder[post_r]]
            left_size = root_index - in_l
            right_size = in_r - root_index
            root = TreeNode(postorder[post_r])
            # get left node
            root.left = dfs(post_l,post_l+left_size-1,in_l,root_index-1)
            # get right node
            root.right = dfs(post_l+left_size,post_l+left_size+right_size-1,root_index+1,in_r)
            return root
        return dfs(0,len(inorder)-1,0,len(inorder)-1)

```
## [889. 根据前序和后序遍历构造二叉树](https://leetcode.cn/problems/construct-binary-tree-from-preorder-and-postorder-traversal/)
首先说明，如果只知道前序遍历和后序遍历，这棵二叉树不一定是唯一
如果二叉树的每个非叶节点都有两个儿子，知道前序和后序就能唯一确定这棵二叉树。
已知道preorder 的第一个节点是root，第一个节点+1是左子树的头节点。
通过这种方式，在postorder里就可以找到左子树的大小和右子树的大小

```python
class Solution:
    def constructFromPrePost(self, preorder: List[int], postorder: List[int]) -> Optional[TreeNode]:
        index = {x: i for i, x in enumerate(postorder)}

        # 注意 post_r 可以省略
        def dfs(pre_l: int, pre_r: int, post_l: int) -> Optional[TreeNode]:
            if pre_l == pre_r:  # 空节点
                return None
            if pre_l + 1 == pre_r:  # 叶子节点
                return TreeNode(preorder[pre_l])
            left_size = index[preorder[pre_l + 1]] - post_l + 1  # 左子树的大小
            left = dfs(pre_l + 1, pre_l + 1 + left_size, post_l)
            right = dfs(pre_l + 1 + left_size, pre_r, post_l + left_size)
            return TreeNode(preorder[pre_l], left, right)

        return dfs(0, len(preorder), 0)  # 左闭右开区间
```


# [235. 二叉搜索树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-search-tree/)

只要这个root 比一个大比一个小或者等于某一个，那他就是公共祖先
如果比两个都大，那就右子树，比两个都小往左子树


```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        x = root.val
        if p.val < x and q.val < x:  # p 和 q 都在左子树
            return self.lowestCommonAncestor(root.left, p, q)
        if p.val > x and q.val > x:  # p 和 q 都在右子树
            return self.lowestCommonAncestor(root.right, p, q)
        return root  # 其它
```


# [236. 二叉树的最近公共祖先](https://leetcode.cn/problems/lowest-common-ancestor-of-a-binary-tree/)

一直往下看，如果遇到自己就返回自己，如果遇到none就返回none，如果左右子树都是none，返回none，如果左右子树有一个不是none，返回不是none的值，如果都不是none，返回自己。

如果首先遇到自己，说明自己可能是两个节点里的公共祖先，也有可能不是，但这个时候不需要往下看了

如果返回非空值，说明左子树/右子树遇到了自己/找到了他们的公共节点。二个都不是none，说明自己就是最近公共祖先。一个none，一个不是none，说明自己只是某一个节点的公共祖先
```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if root in (None, p, q):  # 找到 p 或 q 就不往下递归了，原因见上面答疑
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:  # 左右都找到
            return root  # 当前节点是最近公共祖先
        # 如果只有左子树找到，就返回左子树的返回值
        # 如果只有右子树找到，就返回右子树的返回值
        # 如果左右子树都没有找到，就返回 None（注意此时 right = None）
        return left or right
```



# [621. 任务调度器](https://leetcode.cn/problems/task-scheduler/)

贪心算法，但自己算法写的一坨狗屎。优先移除个数最多的任务，这样才可以用其他个数更少的任务，填补造成的idle。
```python
import heapq
from collections import Counter
from typing import List

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        cnt = Counter(tasks)
        # 用最大堆（Python 默认小堆，取负数模拟）
        heap = [-v for v in cnt.values()]
        heapq.heapify(heap)
        
        time = 0
        while heap:
            temp = []
            # 一轮最多执行 n+1 个任务
            for _ in range(n + 1):
                if heap:
                    x = heapq.heappop(heap)
                    x += 1  # 因为是负的，+1 表示执行一次
                    if x < 0:
                        temp.append(x)
                time += 1
                if not heap and not temp:
                    break
            # 重新入堆
            for item in temp:
                heapq.heappush(heap, item)
        return time
```

# [110. 平衡二叉树](https://leetcode.cn/problems/balanced-binary-tree/)

用-1标志错误
```python
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def get_height(node: Optional[TreeNode]) -> int:
            if node is None:
                return 0
            left_h = get_height(node.left)
            right_h = get_height(node.right)
            if left_h == -1 or right_h == -1 or abs(left_h - right_h) > 1:
                return -1
            return max(left_h, right_h) + 1
        return get_height(root) != -1
```



# [238. 除自身以外数组的乘积](https://leetcode.cn/problems/product-of-array-except-self/)


answer[i] 等于 nums 中除了 nums[i] 之外其余各元素的乘积。换句话说，如果知道了 i 左边所有数的乘积，以及 i 右边所有数的乘积，就可以算出 answer[i]。

于是：

定义 pre[i] 表示从 nums[0] 到 nums[i−1] 的乘积。
定义 suf[i] 表示从 nums[i+1] 到 nums[n−1] 的乘积。
我们可以先计算出从 nums[0] 到 nums[i−2] 的乘积 pre[i−1]，再乘上 nums[i−1]，就得到了 pre[i]，即

pre[i]=pre[i−1]⋅nums[i−1]
同理有

suf[i]=suf[i+1]⋅nums[i+1]
初始值：pre[0]=suf[n−1]=1。按照上文的定义，pre[0] 和 suf[n−1] 都是空子数组的元素乘积，我们规定这是 1，因为 1 乘以任何数 x 都等于 x，这样可以方便递推计算 pre[1]，suf[n−2] 等。

算出 pre 数组和 suf 数组后，有

answer[i]=pre[i]⋅suf[i]

感觉没想到这一步就做不出来，没意识到这一点


```python
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        suf = [1] * n
        for i in range(n - 2, -1, -1):
            suf[i] = suf[i + 1] * nums[i + 1]

        pre = 1
        for i, x in enumerate(nums):
            # 此时 pre 为 nums[0] 到 nums[i-1] 的乘积，直接乘到 suf[i] 中
            suf[i] *= pre
            pre *= x

        return suf
```



# [973. 最接近原点的 K 个点](https://leetcode.cn/problems/k-closest-points-to-origin/)


堆的事
```python
class Solution:
    def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
        q = [(-x ** 2 - y ** 2, i) for i, (x, y) in enumerate(points[:k])]
        heapq.heapify(q)
        
        n = len(points)
        for i in range(k, n):
            x, y = points[i]
            dist = -x ** 2 - y ** 2
            heapq.heappushpop(q, (dist, i))
        
        ans = [points[identity] for (_, identity) in q]
        return ans
```


# [121. 买卖股票的最佳时机](https://leetcode.cn/problems/best-time-to-buy-and-sell-stock/)

dp，定义好持有和不持有两个状态就好了，只能买一次，所以是不持有和第一次持有。状态的变化是

持有  = Max（ 前一天持有  ， 前一天不持有=0 -price -> = -price）->  因为只能买卖一次，所以不能从不持有再变成持有,所以前一天不持有一定要是0
不持有 = Max （前一天不持有， 前一天持有+price）

如果可以买多次的话，

dp是当天持有 和 不持有的最大值

持有  = Max（ 前一天持有  ， 前一天不持有 -price）
不持有 = Max （前一天不持有， 前一天持有+price）

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        @cache  # 缓存装饰器，避免重复计算 dfs 的结果
        def dfs(i: int, hold: bool) -> int:
            if i < 0:
                return -inf if hold else 0
            if hold:
                return max(dfs(i - 1, True), - prices[i])
            return max(dfs(i - 1, False), dfs(i - 1, True) + prices[i])
        return dfs(n - 1, False)
```


# [125. 验证回文串](https://leetcode.cn/problems/valid-palindrome/)
双指针，只有function的调用

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


# [127. 单词接龙](https://leetcode.cn/problems/word-ladder/)


首先题目中并没有给出点与点之间的连线，而是要我们自己去连，条件是字符只能差一个，所以判断点与点之间的关系，要自己判断是不是差一个字符，如果差一个字符，那就是有链接。

然后就是求起点和终点的最短路径长度，这里无向图求最短路，广搜最为合适，广搜只要搜到了终点，那么一定是最短的路径。因为广搜就是以起点中心向四周扩散的搜索。

本题如果用深搜，会非常麻烦。

另外需要有一个注意点：

本题是一个无向图，需要用标记位，标记着节点是否走过，否则就会死循环！
本题给出集合是数组型的，可以转成set结构，查找更快一些


其实时间复杂度没有想象中的
```python
from collections import deque
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        wordSet = set(wordList)
        if len(wordSet) == 0 or endWord not in wordSet:
            return 0
        
        #记录word是否访问过
        visitMap = dict()
        #初始化队列
        queue = deque()
        queue.append(beginWord)
        #初始化visitMap
        visitMap[beginWord] = 1
        
        while queue:
            word = queue.popleft()
            #path为当前word的路径长度
            path = visitMap[word]
            for i in range(len(word)):
                #用一个新单词替换word，因为每次置换一个字母
                word_list = list(word)
                for j in range(26):
                    word_list[i] = chr(ord('a') + j)
                    newWord = ''.join(word_list)
                    #找到endWord，返回path+1
                    if newWord == endWord:
                        return path + 1
                    #如果wordSet中出现了newWord而且newWord没有访问过
                    if newWord in wordSet and newWord not in visitMap:
                        #添加访问信息
                        visitMap[newWord] = path + 1
                        queue.append(newWord)
        return 0
```



# [135. 分发糖果](https://leetcode.cn/problems/candy/)

我们先找从左到右满足最少的糖果，再找从右到左的，最后取两边都满足的值(就是最大值)。

这个是更简单的思路
https://leetcode.cn/problems/candy/solutions/3691236/fen-zu-xun-huan-ba-kun-nan-ti-bian-cheng-fo15/?envType=study-plan-v2&envId=top-interview-150

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        ans = n = len(ratings)  # 先给每人分一个
        i = 0
        while i < n:
            start = i - 1 if i > 0 and ratings[i - 1] < ratings[i] else i

            # 找严格递增段
            while i + 1 < n and ratings[i] < ratings[i + 1]:
                i += 1
            top = i  # 峰顶

            # 找严格递减段
            while i + 1 < n and ratings[i] > ratings[i + 1]:
                i += 1

            inc = top - start  # start 到 top 严格递增
            dec = i - top      # top 到 i 严格递减
            ans += (inc * (inc - 1) + dec * (dec - 1)) // 2 + max(inc, dec)
            i += 1
        return ans
```
# [42. 接雨水](https://leetcode.cn/problems/trapping-rain-water/)



# [13. 罗马数字转整数](https://leetcode.cn/problems/roman-to-integer/)

本题的难点在于处理六种特殊规则，但这六种特殊规则其实可以统一起来：

设 x=s[i−1], y=s[i]，这是两个相邻的罗马数字。
如果 x 的数值小于 y 的数值，那么 x 的数值要取相反数。例如 IV 中的 I 相当于 −1，CM 中的 C 相当于 −100。

```python
# 单个罗马数字到整数的映射
ROMAN = {
    'I': 1,
    'V': 5,
    'X': 10,
    'L': 50,
    'C': 100,
    'D': 500,
    'M': 1000,
}

class Solution:
    def romanToInt(self, s: str) -> int:
        ans = 0
        for x, y in pairwise(s):  # 遍历 s 中的相邻字符
            x, y = ROMAN[x], ROMAN[y]
            # 累加 x 或者 -x，这里 y 只是用来辅助判断 x 的正负
            ans += x if x >= y else -x
        return ans + ROMAN[s[-1]]  # 加上最后一个罗马数字
```

# [12. 整数转罗马数字](https://leetcode.cn/problems/integer-to-roman/)
num 拆分成千位数、百位数、十位数和个位数，分别用罗马数字表示。

```python
R = (
    ("", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"),  # 个位
    ("", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"),  # 十位
    ("", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"),  # 百位
    ("", "M", "MM", "MMM"),  # 千位
)

class Solution:
    def intToRoman(self, num: int) -> str:
        return R[3][num // 1000] + R[2][num // 100 % 10] + R[1][num // 10 % 10] + R[0][num % 10]
```
# [58. 最后一个单词的长度](https://leetcode.cn/problems/length-of-last-word/)

从后往前遍历, 从不是空格开始便利

```python
class Solution:
    def lengthOfLastWord(self, s: str) -> int:
        start = False
        cnt = 0
        for i in range(len(s)-1, -1, -1):
            ch = s[i]
            if not start and ch != ' ':
                cnt = 1
                start = True
            elif start and ch != ' ':
                cnt += 1
            elif start and ch == ' ':
                return cnt
        return cnt
```


# [14. 最长公共前缀](https://leetcode.cn/problems/longest-common-prefix/)

简单题，简单做就好了，要意识到枚举第一个s0的数据
```python
class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        s0 = strs[0]
        for j, c in enumerate(s0):  # 从左到右
            for s in strs:  # 从上到下
                if j == len(s) or s[j] != c:  # 这一列有字母缺失或者不同
                    return s0[:j]  # 0 到 j-1 是公共前缀
        return s0

```