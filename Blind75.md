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


#  [143. 重排链表](https://leetcode.cn/problems/reorder-list/)

问：循环条件 while head2.next 为什么不能写成 while head2？

答：如果链表长度为偶数，例如链表由 [1,2,3,4] 四个节点组成，那么找中间节点并反转后，我们得到的两个链表分别为 head=[1,2,3] 和 head 2 =[4,3]。注意它俩都包含节点 3，具体请看视频 06:01 处的图。如果写成 while head2，会导致最后一轮循环中 3 指向它自己。


```python
class Solution:
    # 876. 链表的中间结点
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        return slow

    # 206. 反转链表
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        pre, cur = None, head
        while cur:
            nxt = cur.next
            cur.next = pre
            pre = cur
            cur = nxt
        return pre

    def reorderList(self, head: Optional[ListNode]) -> None:
        mid = self.middleNode(head)
        head2 = self.reverseList(mid)
        while head2.next:
            nxt = head.next
            nxt2 = head2.next
            head.next = head2
            head2.next = nxt
            head = nxt
            head2 = nxt2
```


# [271. Encode and Decode Strings](https://leetcode.com/problems/encode-and-decode-strings/)

只用//作转义字符不行,不能避免歧义
```python
 class Codec:
    def encode(self, strs):
        """
        Encodes a list of strings to a single string.

        :param strs: List of strings to encode.
        :return: Encoded string.
        """
        # Initialize an empty string to hold the encoded strings
        encoded_string = ''

        # Iterate over each string in the input list
        for s in strs:
            # Replace each occurrence of '/' with '//'
            # This is our way of "escaping" the slash character
            # Then add our delimiter '/:' to the end
            encoded_string += s.replace('/', '//') + '/:'

        # Return the final encoded string
        return encoded_string

    def decode(self, s):
        """
        Decodes a single string to a list of strings.

        :param s: String to decode.
        :return: List of decoded strings.
        """
        # Initialize an empty list to hold the decoded strings
        decoded_strings = []

        # Initialize a string to hold the current string being built
        current_string = ""

        # Initialize an index 'i' to start of the string
        i = 0

        # Iterate while 'i' is less than the length of the encoded string
        while i < len(s):
            # If we encounter the delimiter '/:'
            if s[i:i+2] == '/:':
                # Add the current_string to the list of decoded_strings
                decoded_strings.append(current_string)

                # Clear current_string for the next string
                current_string = ""

                # Move the index 2 steps forward to skip the delimiter
                i += 2

            # If we encounter an escaped slash '//'
            elif s[i:i+2] == '//':
                # Add a single slash to the current_string
                current_string += '/'

                # Move the index 2 steps forward to skip the escaped slash
                i += 2

            # Otherwise, just add the character to current_string
            else:
                current_string += s[i]
                i += 1

        # Return the list of decoded strings
        return decoded_strings
```




# [19. 删除链表的倒数第 N 个结点](https://leetcode.cn/problems/remove-nth-node-from-end-of-list/)

删除节点都要考虑到dummy node
```python
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        """
        0 1 2 3 4 5
        2 steps  -> 2
        then go until null, go 3 step
        dummy to 3, fast to 5
        """
        dummy = ListNode(0)
        dummy.next = head
        fast = dummy
        slow = dummy
        for _ in range(n):
            fast = fast.next
        while fast.next != None:
            fast = fast.next
            slow = slow.next
        # slow in front of the last one
        slow.next = slow.next.next
        return dummy.next
```


# [20. 有效的括号](https://leetcode.cn/problems/valid-parentheses/) 

参考glind75， 用栈


```python
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
# [21. 合并两个有序链表](https://leetcode.cn/problems/merge-two-sorted-lists/)
不知道头是哪个，可以用dummy作为头部，代码更简单易懂

```python
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
# [23. 合并 K 个升序链表](https://leetcode.cn/problems/merge-k-sorted-lists/)

要排序k个，用heap，i是为了防止直接比较对象
```python
import heapq

def mergeKLists(self, lists):
    dummy = ListNode(0)
    cur = dummy
    heap = []

    # 把每个链表的头节点放入堆
    for i, node in enumerate(lists):
        if node:  # 只放非空链表
            heapq.heappush(heap, (node.val, i, node))

    # 不断从堆中弹出最小节点
    while heap:
        val, i, node = heapq.heappop(heap)
        cur.next = node
        cur = cur.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

# [153. 寻找旋转排序数组中的最小值](https://leetcode.cn/problems/find-minimum-in-rotated-sorted-array/)

判断left 和 right 分别在最小值的哪一侧
```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left = 0
        right = len(nums) - 1
        while left <= right:
            mid = (left+right) // 2
            if nums[mid] > nums[-1]:
                left = mid + 1
            else:
                right = mid -1
        return nums[left]

```
# [33. 搜索旋转排序数组](https://leetcode.cn/problems/search-in-rotated-sorted-array/)

同上题，懒得写了，但就是一定要确定mid什么时候再目标值的左侧。确定好一侧之后，另外一侧就简单了
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


# [417. 太平洋大西洋水流问题](https://leetcode.cn/problems/pacific-atlantic-water-flow/)

```python
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights or not heights[0]:
            return []
        m, n = len(heights), len(heights[0])

        pac = [[False] * n for _ in range(m)]
        atl = [[False] * n for _ in range(m)]
        dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        def dfs(r, c, vis):
            if vis[r][c]:
                return        # 已经访问过，直接退出
            vis[r][c] = True
            for dr, dc in dirs:
                nr, nc = r + dr, c + dc
                if 0 <= nr < m and 0 <= nc < n \
                   and not vis[nr][nc] \
                   and heights[nr][nc] >= heights[r][c]:
                    dfs(nr, nc, vis)

        # 太平洋：上边 + 左边
        for i in range(m):
            dfs(i, 0, pac)
        for j in range(n):
            dfs(0, j, pac)

        # 大西洋：下边 + 右边
        for i in range(m):
            dfs(i, n - 1, atl)
        for j in range(n):
            dfs(m - 1, j, atl)

        res = []
        for i in range(m):
            for j in range(n):
                if pac[i][j] and atl[i][j]:
                    res.append([i, j])
        return res

```



# [295. 数据流的中位数](https://leetcode.cn/problems/find-median-from-data-stream/)


要先插入，再弹出，再插入，这样子才能可以保证向对顺序


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


# [424. 替换后的最长重复字符](https://leetcode.cn/problems/longest-repeating-character-replacement/)
仅仅是替换或者统计个数，不一定要考虑使用dp，能滑动窗口就滑动窗口解决

```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        num = [0] * 26
        n = len(s)
        maxn = left = right = 0

        while right < n:
            num[ord(s[right]) - ord("A")] += 1
            maxn = max(maxn, num[ord(s[right]) - ord("A")])
            if right - left + 1 - maxn > k:
                num[ord(s[left]) - ord("A")] -= 1
                left += 1
            right += 1
        
        return right - left
```

# [297. 二叉树的序列化与反序列化](https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/)

看grind75的吧


# [300. 最长递增子序列](https://leetcode.cn/problems/longest-increasing-subsequence/)
还可以继续优化的

```java
    class Solution {
        public int lengthOfLIS(int[] nums) {
            int [] dp = new int[nums.length];
            Arrays.fill(dp,1);
            int res = 1;
            for (int i = 1; i < nums.length; i++) {
                for (int j = i - 1; j >= 0; j--) {
                    if (nums[i] > nums[j]){
                        dp[i] = Math.max(dp[i],dp[j]+1);
                        res = Math.max(res, dp[i]);
                    }
                }
            }
            return res;
        }
    }
```




# [48. 旋转图像](https://leetcode.cn/problems/rotate-image/)

可以看leetcode题解，一定要用方法2
https://leetcode.cn/problems/rotate-image/description/

```java
class Solution {
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        for (int i = 0; i < n / 2; ++i) {
            for (int j = 0; j < (n + 1) / 2; ++j) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[n - j - 1][i];
                matrix[n - j - 1][i] = matrix[n - i - 1][n - j - 1];
                matrix[n - i - 1][n - j - 1] = matrix[j][n - i - 1];
                matrix[j][n - i - 1] = temp;
            }
        }
    }
}
```


# [49. 字母异位词分组](https://leetcode.cn/problems/group-anagrams/)

```java
class Solution {
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<String, List<String>>();
        for (String str : strs) {
            char[] array = str.toCharArray();
            Arrays.sort(array);
            String key = new String(array);
            List<String> list = map.getOrDefault(key, new ArrayList<String>());
            list.add(str);
            map.put(key, list);
        }
        return new ArrayList<List<String>>(map.values());
    }
}
```



# [435. 无重叠区间](https://leetcode.cn/problems/non-overlapping-intervals/)

# [53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/)

# [54. 螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/)

# [55. 跳跃游戏](https://leetcode.cn/problems/jump-game/)
# [572. 另一棵树子树](https://leetcode.cn/problems/subtree-of-another-tree/)

需要根据高度判断是否相同的节点，并不easy。如果要优化时间复杂度的话，并不easy


```python
class Solution:
    # 代码逻辑同 104. 二叉树的最大深度
    def getHeight(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0
        left_h = self.getHeight(root.left)
        right_h = self.getHeight(root.right)
        return max(left_h, right_h) + 1

    # 100. 相同的树
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p is None or q is None:
            return p is q  # 必须都是 None
        return p.val == q.val and \
            self.isSameTree(p.left, q.left) and \
            self.isSameTree(p.right, q.right)

    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        hs = self.getHeight(subRoot)

        # 返回 node 的高度，以及是否找到了 subRoot
        def dfs(node: Optional[TreeNode]) -> (int, bool):
            if node is None:
                return 0, False
            left_h, left_found = dfs(node.left)
            right_h, right_found = dfs(node.right)
            if left_found or right_found:
                return 0, True
            node_h = max(left_h, right_h) + 1
            return node_h, node_h == hs and self.isSameTree(node, subRoot)
        return dfs(root)[1]
```



# [100. 相同的树](https://leetcode.cn/problems/same-tree/)
```python
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p is None or q is None:
            return p is q  # 必须都是 None
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```