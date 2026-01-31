# [773. 滑动谜题](https://leetcode.cn/problems/sliding-puzzle/)
用BFS，找到最终的结果位置，相比于回溯算法，BFS可以找到最短路径
![](https://labuladong.online/images/algo/sliding_puzzle/3.jpeg)

对于这道题，我们抽象出来的图结构也是会包含环的，所以需要一个 visited 数组记录已经走过的节点，避免成环导致死循环。

比如说我从 [\[2,4,1],[5,0,3]] 节点开始，数字 0 向右移动得到新节点 [\[2,4,1],[5,3,0]]，但是这个新节点中的 0 也可以向左移动的，又会回到 [\[2,4,1],[5,0,3]]，这其实就是成环。我们也需要一个 visited 哈希集合来记录已经走过的节点，防止成环导致的死循环。

为了要找到邻居，我们还要记录他的邻居


```python
from collections import deque

class Solution:
    def slidingPuzzle(self, board):
        target = "123450"
        # 将 2x3 的数组转化成字符串作为 BFS 的起点
        start = ""
        for i in range(len(board)):
            for j in range(len(board[0])):
                start += str(board[i][j])
        
        # ****** BFS 算法框架开始 ******
        q = deque()
        visited = set()
        # 从起点开始 BFS 搜索
        q.append(start)
        visited.add(start)
        
        step = 0
        while q: # <extend up -200>![](/images/algo/sliding_puzzle/3.jpeg) #
            # 当前层的节点数量
            sz = len(q)
            for _ in range(sz):
                cur = q.popleft()
                # 判断是否达到目标局面
                if cur == target:
                    return step
                # 将数字 0 和相邻的数字交换位置
                for neighbor_board in self.getNeighbors(cur):
                    # 防止走回头路
                    if neighbor_board not in visited:
                        q.append(neighbor_board)
                        visited.add(neighbor_board)
            step += 1
        # ****** BFS 算法框架结束 ******
        return -1

    def getNeighbors(self, board):
        # 记录一维字符串的相邻索引
        mapping = [
            [1, 3],
            [0, 4, 2],
            [1, 5],
            [0, 4],
            [3, 1, 5],
            [4, 2]
        ] # <extend up -200>![](/images/algo/sliding_puzzle/4.jpeg) #
        
        idx = board.index('0')
        neighbors = []
        for adj in mapping[idx]:
            new_board = self.swap(board, idx, adj)
            neighbors.append(new_board)
        return neighbors

    def swap(self, board, i, j):
        chars = list(board)
        chars[i], chars[j] = chars[j], chars[i]
        return ''.join(chars)
```



#  [752. 打开转盘锁](https://leetcode.cn/problems/open-the-lock/)

穷举所有的困难，排除掉不可能的路，记录会成环的路


1、会走回头路，我们可以从 "0000" 拨到 "1000"，但是等从队列拿出 "1000" 时，还会拨出一个 "0000"，这样的话会产生死循环。

这个问题很好解决，其实就是成环了嘛，我们用一个 visited 集合记录已经穷举过的密码，再次遇到时，不要再加到队列里就行了。

2、没有对 deadends 进行处理，按道理这些「死亡密码」是不能出现的。

这个问题也好处理，额外用一个 deadends 集合记录这些死亡密码，凡是遇到这些密码，不要加到队列里就行了。

或者还可以更简单一些，直接把 deadends 中的死亡密码作为 visited 集合的初始元素，这样也可以达到目的。

下面是完整的代码实现：

```python
class Solution:
    def openLock(self, deadends: List[str], target: str) -> int:
        # 记录需要跳过的死亡密码
        deads = set(deadends)
        if "0000" in deads:
            return -1

        # 记录已经穷举过的密码，防止走回头路
        visited = set()
        q = collections.deque()
        # 从起点开始启动广度优先搜索
        step = 0
        q.append("0000")
        visited.add("0000")
        
        while q:
            sz = len(q)
            # 将当前队列中的所有节点向周围扩散
            for _ in range(sz):
                cur = q.popleft()
                
                # 判断是否到达终点
                if cur == target:
                    return step
                
                # 将一个节点的合法相邻节点加入队列
                for neighbor in self.getNeighbors(cur):
                    if neighbor not in visited and neighbor not in deads:
                        q.append(neighbor)
                        visited.add(neighbor)
            
            # 在这里增加步数
            step += 1
        
        # 如果穷举完都没找到目标密码，那就是找不到了
        return -1

    # 将 s[j] 向上拨动一次
    def plusOne(self, s: str, j: int) -> str:
        ch = list(s)
        if ch[j] == '9':
            ch[j] = '0'
        else:
            ch[j] = chr(ord(ch[j]) + 1)
        return ''.join(ch)

    # 将 s[i] 向下拨动一次
    def minusOne(self, s: str, j: int) -> str:
        ch = list(s)
        if ch[j] == '0':
            ch[j] = '9'
        else:
            ch[j] = chr(ord(ch[j]) - 1)
        return ''.join(ch)

    # 将 s 的每一位向上拨动一次或向下拨动一次，8 种相邻密码
    def getNeighbors(self, s: str) -> List[str]:
        neighbors = []
        for i in range(4):
            neighbors.append(self.plusOne(s, i))
            neighbors.append(self.minusOne(s, i))
        return neighbors


```


# [919. 完全二叉树插入器](https://leetcode.cn/problems/complete-binary-tree-inserter/)
维护好队列的底部
            if cur.right is None or cur.left is None:
                # 找到完全二叉树底部可以进行插入的节点
                self.q.put(cur)
看代码学习怎么维护
```python
class CBTInserter:

    def __init__(self, root: TreeNode):
        self.root = root
        self.candidate = deque()

        q = deque([root])
        while q:
            node = q.popleft()
            if node.left:
                q.append(node.left)
            if node.right:
                q.append(node.right)
            if not (node.left and node.right):
                self.candidate.append(node)

    def insert(self, val: int) -> int:
        candidate_ = self.candidate

        child = TreeNode(val)
        node = candidate_[0]
        ret = node.val
        
        if not node.left:
            node.left = child
        else:
            node.right = child
            candidate_.popleft()
        
        candidate_.append(child)
        return ret

    def get_root(self) -> TreeNode:
        return self.root
```

# [841. 钥匙和房间](https://leetcode.cn/problems/keys-and-rooms/)



# [433. 最小基因变化](https://leetcode.cn/problems/minimum-genetic-mutation/)



# [1926. 迷宫中离入口最近的出口](https://leetcode.cn/problems/nearest-exit-from-entrance-in-maze/)



# [1091. 二进制矩阵中的最短路径](https://leetcode.cn/problems/shortest-path-in-binary-matrix/)



#  [994. 腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/)



# [721. 账户合并](https://leetcode.cn/problems/accounts-merge/)


# [127. 单词接龙](https://leetcode.cn/problems/word-ladder/)



# [365. 水壶问题](https://leetcode.cn/problems/water-and-jug-problem/)