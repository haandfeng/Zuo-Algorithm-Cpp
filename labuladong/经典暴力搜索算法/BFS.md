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

```python



```