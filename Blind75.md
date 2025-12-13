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

```java
class Solution {
    public int eraseOverlapIntervals(int[][] intervals) {
        Arrays.sort(intervals, (a,b)-> {
            return Integer.compare(a[0],b[0]);
        });
        int count = 1;
        for(int i = 1;i < intervals.length;i++){
            if(intervals[i][0] < intervals[i-1][1]){
                intervals[i][1] = Math.min(intervals[i - 1][1], intervals[i][1]);
                continue;
            }else{
                count++;
            }    
        }
        return intervals.length - count;
    }
}
```

# [53. 最大子数组和](https://leetcode.cn/problems/maximum-subarray/)

```java
class Solution {
        public int maxSubArray(int[] nums) {
            int[] prefixSum = new int[nums.length + 1];

            // 计算前缀和
            for (int i = 1; i <= nums.length; i++) {
                prefixSum[i] = prefixSum[i - 1] + nums[i - 1];
            }

            int preMin = 0;                 // 当前最小前缀和
            int maxSum = Integer.MIN_VALUE; // 最大子数组和

            for (int i = 1; i <= nums.length; i++) {
                // 用当前前缀和减去之前最小前缀和，更新最大值
                maxSum = Math.max(maxSum, prefixSum[i] - preMin);

                // 更新最小前缀和
                preMin = Math.min(preMin, prefixSum[i]);
            }

            return maxSum;
        }
}
```

# [54. 螺旋矩阵](https://leetcode.cn/problems/spiral-matrix/)
```java
class Solution {
        public List<Integer> spiralOrder(int[][] matrix) {
            int left = 0, right = matrix[0].length - 1,up = 0, down = matrix.length - 1;
            List<Integer> res = new ArrayList<>();
            while (left <= right && up <= down) {
                for (int i = left; i <= right; i++) {
                    res.add(matrix[up][i]);
                }
                if(++up > down) {return res;}
                for (int i = up; i <= down; i++) {
                    res.add(matrix[i][right]);
                }
                if(--right < left) {return res;}
                for (int i = right; i >= left; i--) {
                    res.add(matrix[down][i]);
                }
                if(--down < up) {return res;}
                for (int i = down; i >= up; i--) {
                    res.add(matrix[i][left]);
                }
                if(++left > right) {return res;}
            }
            return res;
        }
}
```
# [55. 跳跃游戏](https://leetcode.cn/problems/jump-game/)

```java
class Solution {
    public boolean canJump(int[] nums) {
        if (nums == null || nums.length == 0) return false;
        // 长度为 1，起点就是终点
        if (nums.length == 1) return true;

        int nextRange = nums[0];
        int cur = 0;

        while (true) {
            boolean updateFlag = false;
            // 防止 i 走出数组范围
            int limit = Math.min(nextRange, nums.length - 1);

            for (int i = cur; i <= limit; i++) {
                if (i + nums[i] > nextRange) {
                    nextRange = i + nums[i];
                    cur = i;
                    updateFlag = true;
                }
                // 只要最远能到最后一个下标，就可以直接返回 true
                if (nextRange >= nums.length - 1) {
                    return true;
                }
            }

            // 本轮完全没法扩展范围，说明被卡住了
            if (!updateFlag) {
                return false;
            }
        }
    }
}
```

只要维护最右可达位置就好了，我写的并不优美
```java
class Solution {
    public boolean canJump(int[] nums) {
        int mx = 0;
        for (int i = 0; i < nums.length; i++) {
            if (i > mx) { // 无法到达 i
                return false;
            }
            mx = Math.max(mx, i + nums[i]); // 从 i 最右可以跳到 i + nums[i]
        }
        return true;
    }
}
```


# [57. 插入区间](https://leetcode.cn/problems/insert-interval/)
[[Grind75#[57. 插入区间](https //leetcode.cn/problems/insert-interval/)]]

新写的一个思路，差不多，但就是要注意，要及时把merge好的区间塞回队列里面，不要影响相对顺序。
```java
class Solution {
    public int[][] insert(int[][] intervals, int[] newInterval) {
        List<int[]> res = new ArrayList<>();

        for (int[] cur : intervals) {
            // 1. 当前区间在 newInterval 左边，完全没重叠
            if (cur[1] < newInterval[0]) {
                res.add(cur);
            }
            // 2. 当前区间在 newInterval 右边，完全没重叠
            else if (cur[0] > newInterval[1]) {
                // 先把合并好的 newInterval 放进去
                res.add(newInterval);
                // 再把当前区间当作“新的 newInterval”，后面继续处理
                newInterval = cur;
            }
            // 3. 有重叠，合并到 newInterval 上
            else {
                newInterval[0] = Math.min(newInterval[0], cur[0]);
                newInterval[1] = Math.max(newInterval[1], cur[1]);
            }
        }

        // 循环结束后，把最后一个 newInterval 加进去
        res.add(newInterval);

        return res.toArray(new int[res.size()][]);
    }
}
```
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

# [62. Unique Paths](https://leetcode.com/problems/unique-paths/)

```java
class Solution {
        public int uniquePaths(int m, int n) {
            int[][] dp = new int[m][n];
            dp[0][0] = 1;
            for (int i = 1; i < m; i++) {
                dp[i][0] = 1;
            }
            for (int j = 1; j < n; j++) {
                dp[0][j] = 1;
            }
            for (int i = 1; i < m; i++) {
                for (int j = 1; j < n; j++) {
                    dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                }
            }
            return dp[m - 1][n - 1];
        }
}
```



# [190. 颠倒二进制位](https://leetcode.cn/problems/reverse-bits/)
分成几步看：
	1.	n & 1
	•	取出 n 的最低位（LSB）
	•	结果要么是 0，要么是 1
	2.	(n & 1) << (31 - i)
	•	把这个最低位挪到目标位置：
	•	当前是第 i 次循环：
	•	第 0 次循环：最低位应该放到 最高位 位置 31
	•	第 1 次循环：应该放到 30
	•	…
	•	所以右边 shift 的位置是 31 - i
	3.	rev |= ...
	•	用按位或，把这一个 bit 写进 rev 中对应的位置
	•	因为每一轮写的都是不同位置，所以不会互相覆盖
```java
public class Solution {
    public int reverseBits(int n) {
        int rev = 0;
        for (int i = 0; i < 32 && n != 0; ++i) {
            rev |= (n & 1) << (31 - i);
            n >>>= 1;
        }
        return rev;
    }
}
```


# [191. 位1的个数](https://leetcode.cn/problems/number-of-1-bits/)


```java
    public int hammingWeight(int n) {
        int cnt = 0;
        for(int i = 0; i < 32; ++i){
            if ((n & 1) == 1){
                ++cnt;
            }
            n >>>= 1;
        }
        return cnt;
    }
```

# [70. 爬楼梯](https://leetcode.cn/problems/climbing-stairs/)

```java
public class Solution {
    public int climbStairs(int n) {
        if (n == 1) {
            return 1;
        }
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        return dp[n];
    }
}
```
# [323. Number of Connected Components in an Undirected Graph](https://leetcode.com/problems/number-of-connected-components-in-an-undirected-graph/)

dfs

```java
class Solution {
    
     private void dfs(List<Integer>[] adjList, int[] visited, int startNode) {
        visited[startNode] = 1;
         
        for (int i = 0; i < adjList[startNode].size(); i++) {
            if (visited[adjList[startNode].get(i)] == 0) {
                dfs(adjList, visited, adjList[startNode].get(i));
            }
        }
    }
    
    public int countComponents(int n, int[][] edges) {
        int components = 0;
        int[] visited = new int[n];
        
        List<Integer>[] adjList = new ArrayList[n]; 
        for (int i = 0; i < n; i++) {
            adjList[i] = new ArrayList<Integer>();
        }
        
        for (int i = 0; i < edges.length; i++) {
            adjList[edges[i][0]].add(edges[i][1]);
            adjList[edges[i][1]].add(edges[i][0]);
        }
        
        for (int i = 0; i < n; i++) {
            if (visited[i] == 0) {
                components++;
                dfs(adjList, visited, i);
            }
        }
        return components;
    }
}
```


#  [73. 矩阵置零](https://leetcode.cn/problems/set-matrix-zeroes/)
```java
class Solution {
    public void setZeroes(int[][] matrix) {
        int m = matrix.length;
        int n = matrix[0].length;

        // 记录第一行是否包含 0
        boolean firstRowHasZero = false; 
        for (int x : matrix[0]) {
            if (x == 0) {
                firstRowHasZero = true;
                break;
            }
        }

        // 记录第一列是否包含 0
        boolean firstColHasZero = false; 
        for (int i = 0; i < m; i++) {
            if (matrix[i][0] == 0) {
                firstColHasZero = true;
                break;
            }
        }

        // 用第一列 matrix[i][0] 保存 rowHasZero[i]
        // 用第一行 matrix[0][j] 保存 colHasZero[j]
        for (int i = 1; i < m; i++) { // 无需遍历第一行，如果 matrix[0][j] 本身是 0，那么相当于 colHasZero[j] 已经是 true
            for (int j = 1; j < n; j++) { // 无需遍历第一列，如果 matrix[i][0] 本身是 0，那么相当于 rowHasZero[i] 已经是 true
                if (matrix[i][j] == 0) {
                    matrix[i][0] = 0; // 相当于 rowHasZero[i] = true
                    matrix[0][j] = 0; // 相当于 colHasZero[j] = true
                }
            }
        }

        for (int i = 1; i < m; i++) { // 跳过第一行，留到最后修改
            for (int j = 1; j < n; j++) { // 跳过第一列，留到最后修改
                if (matrix[i][0] == 0 || matrix[0][j] == 0) { // i 行或 j 列有 0
                    matrix[i][j] = 0;
                }
            }
        }

        // 如果第一列一开始就包含 0，那么把第一列全变成 0
        if (firstColHasZero) {
            for (int[] row : matrix) {
                row[0] = 0;
            }
        }

        // 如果第一行一开始就包含 0，那么把第一行全变成 0
        if (firstRowHasZero) {
            Arrays.fill(matrix[0], 0);
        }
    }
}
```



# [76. 最小覆盖子串](https://leetcode.cn/problems/minimum-window-substring/)

上面的代码每次都要花费 `O(|Σ|)` 的时间去判断是否涵盖，能不能优化到 `O(1)` 呢？  
可以。用一个变量 `less` 维护目前子串中有 `less` 种字符的出现次数小于 `t` 中该字符的出现次数。
具体来说（注意下面算法中的 `less` 变量）：
1. 初始化 `ansLeft = -1, ansRight = m`，用来记录最短子串的左右端点，其中 `m` 是 `s` 的长度。
2. 用一个哈希表（或者数组）`cntT` 统计 `t` 中每个字符的出现次数。
3. 初始化 `left = 0`，以及一个空哈希表（或者数组）`cntS`，用来统计 `s` 的当前子串中每个字符的出现次数。
4. 初始化 `less` 为 `t` 中不同字符的个数。
5. 遍历 `s`，设当前枚举的子串右端点为 `right`，把字符 `c = s[right]` 的出现次数加一。  
   加一后，如果 `cntS[c] = cntT[c]`，说明字符 `c` 的出现次数已经满足要求，把 `less` 减一。
6. 如果 `less == 0`，说明 `cntS` 中每个字符及其出现次数都大于等于 `cntT` 中该字符的出现次数，那么：
   a. 如果 `right - left < ansRight - ansLeft`，说明我们找到了更短的子串，更新  
      `ansLeft = left, ansRight = right`。
   b. 把字符 `x = s[left]` 的出现次数减一。减一前，如果 `cntS[x] = cntT[x]`，说明 `x` 的出现次数将不再满足要求，把 `less` 加一。

   c. 左端点右移，即 `left += 1`。

   d. 重复上面三步，直到 `less > 0`，即 `s` 中有字符的出现次数小于 `t` 中该字符的出现次数为止。

7. 最后，如果 `ansLeft < 0`，说明没有找到符合要求的子串，返回空字符串；  
   否则返回下标 `ansLeft` 到下标 `ansRight` 之间的子串。

代码实现时，可以把 `cntS` 和 `cntT` 合并成一个 `cnt`，定义：

```text
cnt[x] := cntT[x] - cntS[x]
```

```java
class Solution {
    public String minWindow(String S, String t) {
        int[] cntS = new int[128]; // s 子串字母的出现次数
        int[] cntT = new int[128]; // t 中字母的出现次数
        for (char c : t.toCharArray()) {
            cntT[c]++;
        }

        char[] s = S.toCharArray();
        int m = s.length;
        int ansLeft = -1;
        int ansRight = m;

        int left = 0;
        for (int right = 0; right < m; right++) { // 移动子串右端点
            cntS[s[right]]++; // 右端点字母移入子串
            while (isCovered(cntS, cntT)) { // 涵盖
                if (right - left < ansRight - ansLeft) { // 找到更短的子串
                    ansLeft = left; // 记录此时的左右端点
                    ansRight = right;
                }
                cntS[s[left]]--; // 左端点字母移出子串
                left++;
            }
        }

        return ansLeft < 0 ? "" : S.substring(ansLeft, ansRight + 1);
    }

    private boolean isCovered(int[] cntS, int[] cntT) {
        for (int i = 'A'; i <= 'Z'; i++) {
            if (cntS[i] < cntT[i]) {
                return false;
            }
        }
        for (int i = 'a'; i <= 'z'; i++) {
            if (cntS[i] < cntT[i]) {
                return false;
            }
        }
        return true;
    }
}
```



# [79. 单词搜索](https://leetcode.cn/problems/word-search/)

没想象中的难，但自己太懒了，没有写

```java
class Solution {
    private static final int[][] DIRS = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};

    public boolean exist(char[][] board, String word) {
        char[] w = word.toCharArray();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[i].length; j++) {
                if (dfs(i, j, 0, board, w)) {
                    return true; // 搜到了！
                }
            }
        }
        return false; // 没搜到
    }

    private boolean dfs(int i, int j, int k, char[][] board, char[] word) {
        if (board[i][j] != word[k]) { // 匹配失败
            return false;
        }
        if (k == word.length - 1) { // 匹配成功！
            return true;
        }
        board[i][j] = 0; // 标记访问过
        for (int[] d : DIRS) {
            int x = i + d[0];
            int y = j + d[1]; // 相邻格子
            if (0 <= x && x < board.length && 0 <= y && y < board[x].length && dfs(x, y, k + 1, board, word)) {
                return true; // 搜到了！
            }
        }
        board[i][j] = word[k]; // 恢复现场
        return false; // 没搜到
    }
}
```


# [207. 课程表](https://leetcode.cn/problems/course-schedule/)

```java
class Solution {
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int[] indegrees = new int[numCourses];
        List<List<Integer>> adjacency = new ArrayList<>();
        Queue<Integer> queue = new LinkedList<>();
        for(int i = 0; i < numCourses; i++)
            adjacency.add(new ArrayList<>());
        // Get the indegree and adjacency of every course.
        for(int[] cp : prerequisites) {
            indegrees[cp[0]]++;
            adjacency.get(cp[1]).add(cp[0]);
        }
        // Get all the courses with the indegree of 0.
        for(int i = 0; i < numCourses; i++)
            if(indegrees[i] == 0) queue.add(i);
        // BFS TopSort.
        while(!queue.isEmpty()) {
            int pre = queue.poll();
            numCourses--;
            for(int cur : adjacency.get(pre))
                if(--indegrees[cur] == 0) queue.add(cur);
        }
        return numCourses == 0;
    }
}
```

# [208. 实现 Trie (前缀树)](https://leetcode.cn/problems/implement-trie-prefix-tree/)

多一个is end 判断自己是否到叶节点
```java
class Trie {
    class Node {
        public char val;
        public Node[] children;
        public boolean isEnd;

        public Node(char val) {
            this.val = val;
            this.children = new Node[26];
            this.isEnd = false;
        }
    }

    public Node root;

    public Trie() {
        root = new Node('\0');
    }

    public void insert(String word) {
        Node cur = root;
        for (char ch : word.toCharArray()) {
            int idx = ch - 'a';
            if (cur.children[idx] == null) {
                cur.children[idx] = new Node(ch);
            }
            cur = cur.children[idx];   // ✅ 无论新旧都要往下走
        }
        cur.isEnd = true;              // ✅ 标记单词结束
    }

    public boolean search(String word) {
        Node cur = root;
        for (char ch : word.toCharArray()) {
            int idx = ch - 'a';
            if (cur.children[idx] == null) return false; // ✅ 遇到断路直接 false
            cur = cur.children[idx];
        }
        return cur.isEnd; // ✅ 必须是完整单词
    }

    public boolean startsWith(String prefix) {
        Node cur = root;
        for (char ch : prefix.toCharArray()) {
            int idx = ch - 'a';
            if (cur.children[idx] == null) return false;
            cur = cur.children[idx];
        }
        return true; // ✅ 只要路径存在即可
    }
}
```



# [338. 比特位计数](https://leetcode.cn/problems/counting-bits/)


对于所有的数字，只有两类：

奇数：二进制表示中，奇数一定比前面那个偶数多一个 1，因为多的就是最低位的 1。
          举例： 
         0 = 0       1 = 1
         2 = 10      3 = 11
偶数：二进制表示中，偶数中 1 的个数一定和除以 2 之后的那个数一样多。因为最低位是 0，除以 2 就是右移一位，也就是把那个 0 抹掉而已，所以 1 的个数是不变的。
           举例：
          2 = 10       4 = 100       8 = 1000
          3 = 11       6 = 110       12 = 1100


```java

```

# 
```java
class WordDictionary {
    class Node {
        Node[] tns = new Node[26];
        boolean isWord;
    }
    Node root = new Node();
    public void addWord(String s) {
        Node p = root;
        for (int i = 0; i < s.length(); i++) {
            int u = s.charAt(i) - 'a';
            if (p.tns[u] == null) p.tns[u] = new Node();
            p = p.tns[u];
        }
        p.isWord = true;
    }
    public boolean search(String s) {
        return dfs(s, root, 0);
    }
    boolean dfs(String s, Node p, int sIdx) {
        int n = s.length();
        if (n == sIdx) return p.isWord;
        char c = s.charAt(sIdx);
        if (c == '.') {
            for (int j = 0; j < 26; j++) {
                if (p.tns[j] != null && dfs(s, p.tns[j], sIdx + 1)) return true;
            }
            return false;
        } else {
            int u = c - 'a';
            if (p.tns[u] == null) return false;
            return dfs(s, p.tns[u], sIdx + 1);
        }
    }
}
```
# [217. 存在重复元素](https://leetcode.cn/problems/contains-duplicate/)

```java
    public boolean containsDuplicate(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int x : nums) {
            set.add(x);
        }
        return set.size() < nums.length;
    }
```

# [100. 相同的树](https://leetcode.cn/problems/same-tree/)
```python
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if p is None or q is None:
            return p is q  # 必须都是 None
        return p.val == q.val and self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```