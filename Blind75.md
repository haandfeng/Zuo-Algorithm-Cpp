https://leetcode.com/problem-list/oq45f3x3/
# [128. 最长连续序列](https://leetcode.cn/problems/longest-consecutive-sequence/)

用哈希表占位置，遍历，找到序列的开始点一个个往前看
```python
class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        st = set(nums)  # 把 nums 转成哈希集合
        ans = 0
        for x in st:  # 遍历哈希集合
            if x - 1 in st:  # 如果 x 不是序列的起点，直接跳过
                continue
            # x 是序列的起点
            y = x + 1
            while y in st:  # 不断查找下一个数是否在哈希集合中
                y += 1
            # 循环结束后，y-1 是最后一个在哈希集合中的数
            ans = max(ans, y - x)  # 从 x 到 y-1 一共 y-x 个数
        return ans
```



# [1. 两数之和](https://leetcode.cn/problems/two-sum/)

看glind

# [3. 无重复字符的最长子串](https://leetcode.cn/problems/longest-substring-without-repeating-characters/)
滑动窗口
看glind
# [5. 最长回文子串](https://leetcode.cn/problems/longest-palindromic-substring/)
看glind
要认真动脑思考。要确定dp数组的含义，怎么通过DP数组的含义，递推到下一步


# [133. 克隆图](https://leetcode.cn/problems/clone-graph/)

见glind


我的也行，我这个是bfs，注意用dict把旧没电

```python
class Solution:
    def cloneGraph(self, node: Optional['Node']) -> Optional['Node']:
        if not node:
            return node
        visited = {}
        queue = deque()
        root = Node(node.val)
        queue.append(node)
        visited[node] = root
        while queue:
            originNode = queue.pop()
            for n in originNode.neighbors:
                if n not in visited:
                    newn = Node(n.val)
                    visited[n] = newn
                    queue.append(n)
                    visited[originNode].neighbors.append(newn)
                else:
                    visited[originNode].neighbors.append(visited[n])
        return root
```
# [261. Graph Valid Tree](https://leetcode.com/problems/graph-valid-tree/)


很多特例没找到，给的所有参数都是有用的，忽略了n
```python
from collections import defaultdict
from typing import List

class Solution:
    def validTree(self, n: int, edges: List[List[int]]) -> bool:
        # 特判：只有一个点且没有边，是一棵树
        if n == 1 and not edges:
            return True
        # 树的必要条件：边数必须是 n - 1
        if len(edges) != n - 1:
            return False

        g = defaultdict(list)
        for a, b in edges:
            g[a].append(b)
            g[b].append(a)

        visited = set()

        def dfs(u, v):
            visited.add(v)
            for nei in g[v]:
                if nei == u:
                    continue
                if nei in visited:
                    return False     # 找到环
                if not dfs(v, nei):  # 递归结果要往外传
                    return False
            return True

        # 从 0 开始 DFS
        if not dfs(-1, 0):
            return False

        # 必须所有节点都被访问到
        return len(visited) == n
```

# [518. 零钱兑换 II](https://leetcode.cn/problems/coin-change-ii/)

完全背包，组合问题，先物品再背包

```python
class Solution:
    def change(self, amount: int, coins: List[int]) -> int:
        f = [1] + [0] * amount
        for x in coins:
            for c in range(x, amount + 1):
                f[c] += f[c - x]
        return f[amount]
```


# [647. 回文子串](https://leetcode.cn/problems/palindromic-substrings/)

从数量上不好推，直接判断每一个是不是回文子串

看动态规划
# [11. 盛最多水的容器](https://leetcode.cn/problems/container-with-most-water/)

双指针，贪心算法
看glind75


# [139. 单词拆分](https://leetcode.cn/problems/word-break/)

一开始用回溯做，路径的拆分不如灵神那么干净利落，导致没想到要用记忆化搜索。不过用记忆化搜索已正常，我原本的思路：
我会不断的看 i +1 重复的看，这斌不好
```python
class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        path = []
        word = set(wordDict)
        def dfs(i: int) -> bool:
            if i == len(s):
                if "".join(path) in word:
                    return True
                else:
                    return False
            path.append(s[i])
            if "".join(path) in word:
                if not dfs(i+1):
                    path.clear()
                    return dfs(i+1)
                else:
                    return True
            else:
                return dfs(i+1)
        return dfs(0)
```


```python

class Solution:
    def wordBreak(self, s: str, wordDict: List[str]) -> bool:
        max_len = max(map(len, wordDict))  # 用于限制下面 j 的循环次数
        words = set(wordDict)  # 便于快速判断 s[j:i] in words

        @cache  # 缓存装饰器，避免重复计算 dfs 的结果（记忆化）
        def dfs(i: int) -> bool:
            if i == 0:  # 成功拆分！
                return True
            for j in range(i - 1, max(i - max_len - 1, -1), -1):
                if s[j:i] in words and dfs(j):
                    return True
            return False

        return dfs(len(s))
```



# [141. 环形链表](https://leetcode.cn/problems/linked-list-cycle/)

看glind75


# [268. 丢失的数字](https://leetcode.cn/problems/missing-number/)

## 原地哈希

事实上，我们可以将 nums 本身作为哈希表进行使用，将 nums[i] 放到其应该出现的位置（下标） nums[i] 上（ nums[i]<n ），然后对 nums 进行检查，找到满足 nums[i] !=i 的位置即是答案，如果不存在 nums[i] !=i 的位置，则 n 为答案。


# 异或
找缺失数、找出现一次数都是异或的经典应用。

我们可以先求得 [1,n] 的异或和 ans，然后用 ans 对各个 nums[i] 进行异或。

这样最终得到的异或和表达式中，只有缺失元素出现次数为 1 次，其余元素均出现两次（x⊕x=0），即最终答案 ans 为缺失元素。

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        xor = 0
        for i, num in enumerate(nums):
            xor ^= i ^ num
        return xor ^ len(nums)
```



# [15. 三数之和](https://leetcode.cn/problems/3sum/)

去重逻辑要自己写一遍才行
```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        ans = []
        for i in range(0,len(nums)-1):
            if i >=1 and nums[i] == nums[i-1]:
                continue
            j = i +1
            k = len(nums) -1
            target = -nums[i]
            while j < k:
                if nums[j] + nums[k] < target:
                    j += 1
                    while j < k  and nums[j] == nums[j-1]:
                        j+=1
                elif nums[j] + nums[k] > target:
                    k-=1
                    while j < k and nums[k + 1] == nums[k]:
                        k-=1
                else:
                    ans.append([nums[i],nums[j],nums[k]])
                    j += 1
                    while j < k  and nums[j] == nums[j-1]:
                        j+=1
        return ans

```