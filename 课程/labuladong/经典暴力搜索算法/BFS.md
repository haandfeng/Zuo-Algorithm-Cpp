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

其实题目输入的就是一个 自 邻接表 形式表示的图。
你心里那棵穷举树结构出来没有？如果没有的话，可以看一下可视化面板，BFS和 DFS 的解法代码可视化我都做了：
```python
class Solution:
    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        n = len(rooms)
        # 记录访问过的房间
        visited = [False] * n
        queue = collections.deque([0])
        # 在队列中加入起点，启动 BFS
        visited[0] = True

        while queue:
            room = queue.popleft()
            for nextRoom in rooms[room]:
                if not visited[nextRoom]:
                    visited[nextRoom] = True
                    queue.append(nextRoom)

        for v in visited:
            if not v:
                return False
        return True
```
# [433. 最小基因变化](https://leetcode.cn/problems/minimum-genetic-mutation/)

BFS就好了，没什么难度
枚举所有可能的情况
```python
class Solution:
    def minMutation(self, startGene: str, endGene: str, bank: List[str]) -> int:
        bankSet = set(bank)
        if endGene not in bankSet:
            return -1
        AGCT = ['A', 'G', 'C', 'T']

        # BFS 标准框架
        q = collections.deque()
        visited = set()
        q.append(startGene)
        visited.add(startGene)
        step = 0
        while q:
            sz = len(q)
            for j in range(sz):
                cur = q.popleft()
                if cur == endGene:
                    return step
                # 向周围扩散
                for newGene in self.getAllMutation(cur):
                    if newGene not in visited and newGene in bankSet:
                        q.append(newGene)
                        visited.add(newGene)
            step += 1
        return -1

    # 当前基因的每个位置都可以变异为 A/G/C/T，穷举所有可能的结构
    def getAllMutation(self, gene: str) -> List[str]:
        res = []
        geneChars = list(gene)
        for i in range(len(geneChars)):
            oldChar = geneChars[i]
            for newChar in ['A', 'G', 'C', 'T']:
                geneChars[i] = newChar
                res.append("".join(geneChars))
            geneChars[i] = oldChar
        return res
```

# [1926. 迷宫中离入口最近的出口](https://leetcode.cn/problems/nearest-exit-from-entrance-in-maze/)

```python
from collections import deque

class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        m = len(maze)
        n = len(maze[0])
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

        # BFS 算法的队列和 visited 数组
        queue = deque()
        visited = [[False for _ in range(n)] for _ in range(m)]
        queue.append(entrance)
        visited[entrance[0]][entrance[1]] = True
        # 启动 BFS 算法从 entrance 开始像四周扩散
        step = 0
        while queue:
            sz = len(queue)
            step += 1
            # 扩散当前队列中的所有节点
            for i in range(sz):
                cur = queue.popleft()
                # 每个节点都会尝试向上下左右四个方向扩展一步
                for dir in dirs:
                    x = cur[0] + dir[0]
                    y = cur[1] + dir[1]
                    if x < 0 or x >= m or y < 0 or y >= n or visited[x][y] or maze[x][y] == '+':
                        continue
                    if x == 0 or x == m - 1 or y == 0 or y == n - 1:
                        # 走到边界（出口）
                        return step
                    visited[x][y] = True
                    queue.append([x, y])
        return -1
```

# [1091. 二进制矩阵中的最短路径](https://leetcode.cn/problems/shortest-path-in-binary-matrix/)

模版
```python
from collections import deque

class Solution:
    def shortestPathBinaryMatrix(self, grid: List[List[int]]) -> int:
        m, n = len(grid), len(grid[0])
        if grid[0][0] == 1 or grid[m - 1][n - 1] == 1:
            return -1

        q = deque()
        # 需要记录走过的路径，避免死循环
        visited = [[False] * n for _ in range(m)]

        # 初始化队列，从 (0, 0) 出发
        q.append((0, 0))
        visited[0][0] = True
        pathLen = 1

        # 执行 BFS 算法框架，从值为 0 的坐标开始向八个方向扩散
        dirs = [
            (0, 1), (0, -1), (1, 0), (-1, 0),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        while q:
            sz = len(q)
            for _ in range(sz):
                x, y = q.popleft()
                if x == m - 1 and y == n - 1:
                    return pathLen
                # 向八个方向扩散
                for dx, dy in dirs:
                    nextX, nextY = x + dx, y + dy
                    # 确保相邻的这个坐标没有越界且值为 0 且之前没有走过
                    if 0 <= nextX < m and 0 <= nextY < n and grid[nextX][nextY] == 0 and not visited[nextX][nextY]:
                        q.append((nextX, nextY))
                        visited[nextX][nextY] = True
            pathLen += 1
        return -1
```
#  [994. 腐烂的橘子](https://leetcode.cn/problems/rotting-oranges/)

简单题
```python
from collections import deque

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        queue = deque()
        m, n = len(grid), len(grid[0])
        # 把所有腐烂的橘子加入队列，作为 BFS 的起点
        for i in range(m):
            for j in range(n):
                if grid[i][j] == 2:
                    queue.append((i, j))
        
        # 方向数组，方便计算上下左右的坐标
        dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # BFS 算法框架
        step = 0
        while queue:
            sz = len(queue)
            # 取出当前层所有节点，往四周扩散一层
            for _ in range(sz):
                point = queue.popleft()
                for dir in dirs:
                    x, y = point[0] + dir[0], point[1] + dir[1]
                    if 0 <= x < m and 0 <= y < n and grid[x][y] == 1:
                        grid[x][y] = 2
                        queue.append((x, y))
            # 扩散步数加一
            step += 1

        # 检查是否还有新鲜橘子
        for i in range(m):
            for j in range(n):
                # 有新鲜橘子，返回 -1
                if grid[i][j] == 1:
                    return -1

        # 注意，BFS 扩散的步数需要减一才是最终结果
        # 你可以用最简单的情况，比方说只有一个腐烂橘子的情况验证一下
        return step - 1 if step else 0
```
# [721. 账户合并](https://leetcode.cn/problems/accounts-merge/)
人的名字是没办法唯一标识一个人的，所以一定是用邮箱号标识一个人。

并查集直接是人就好了，如果这个邮箱ID号出现过（用一个map统计），就把那两个人连在一起

最后每个邮箱号通过一个map，找到对应的人（UF）找，然后重新翻到一个map里（人ID），所有邮箱，排序，就好了

如果通过dfs找联通分量的话
邮箱对应一个id list

遍历的时候遍历id，通过id找到邮箱，通过邮箱再找回id，dfs（id），通过这种方式遍历所有id
# [127. 单词接龙](https://leetcode.cn/problems/word-ladder/)

简单bfs 26叉树

```python
class Solution:
    def ladderLength(self, beginWord: str, endWord: str, wordList: List[str]) -> int:
        # 将 wordList 转换为 HashSet，加速查找
        wordSet = set(wordList)
        if endWord not in wordSet:
            return 0

        # 直接套用 BFS 算法框架
        q = collections.deque([beginWord])
        visited = set([beginWord])
        step = 1
        while q:
            sz = len(q)
            for i in range(sz):
                # 穷举 curWord 修改一个字符能得到的单词
                # 即对每个字符，穷举 26 个字母
                curWord = q.popleft()
                chars = list(curWord)
                # 开始穷举每一位字符 curWord[j]
                for j in range(len(curWord)):
                    originChar = chars[j]
                    # 对每一位穷举 26 个字母
                    for c in 'abcdefghijklmnopqrstuvwxyz':
                        if c == originChar:
                            continue
                        chars[j] = c
                        # 如果构成的新单词在 wordSet 中，就是找到了一个可行的下一步
                        newWord = ''.join(chars)
                        if newWord in wordSet and newWord not in visited:
                            if newWord == endWord:
                                return step + 1
                            q.append(newWord)
                            visited.add(newWord)
                    # 最后别忘了把 curWord[j] 恢复
                    chars[j] = originChar
            # 这里增加步数
            step += 1
        return 0
```

# [365. 水壶问题](https://leetcode.cn/problems/water-and-jug-problem/)


基本思路
这道题非常经典，也非常有意思。我没记错的话，我小学就做过类似的数学题，不过那时候也没什么章法，反正不断地倒来倒去肯定能蒙出目标水量。
那么到了现在这个阶段，根据我之前介绍的 自 算法学习心法，我们首先想的应该是把问题抽象成树结构，然后穷举所有的倒法，看看是否有可能凑出 targetCapacity。
如果把两个桶中现有的水量作为「状态」，那么题目给出的几种倒水方法就是导致「状态」发生改变的「选择」，这样一来，你完全可以用 自 动态规划详解进阶篇 里讲过的动态规划思路来做。
同时，你也可以用自 BFS 算法框架 来解决这道题。我这里就写 BFS 算法吧，具体细节看代码中的注释。
最后，这道题的最优解法是数学方法，你可以去了解一下「裴蜀定理」，也叫「贝祖定理」，有兴趣的读者可以自行搜索，我这里只给出最通用的计算机算法思路，不展开讲数学方法了。


这里的状态管理主要是针对其他语言的，其实对于python，tuble是可以哈西的，不需要这样操作
```python
class Solution:

    def canMeasureWater(self, jug1Capacity: int, jug2Capacity: int, targetCapacity: int) -> bool:
        # BFS 算法的队列
        q = collections.deque()
        # 用来记录已经遍历过的状态，把元组转化成数字方便存储哈希集合
        # 转化方式是 (x, y) -> (x * (jug2Capacity + 1) + y)，和二维数组坐标转一维坐标是一样的原理
        # 因为水桶 2 的取值是 [0, jug2Capacity]，所以需要额外加一，请类比二维数组坐标转一维坐标
        # 且考虑到题目输入的数据规模较大，相乘可能导致 int 溢出，所以使用 long 类型
        visited = set()
        # 添加初始状态，两个桶都没有水
        q.append((0, 0))
        visited.add(0 * (jug2Capacity + 1) + 0)

        while q:
            curState = q.popleft()
            if (curState[0] == targetCapacity or curState[1] == targetCapacity
                    or curState[0] + curState[1] == targetCapacity):
                # 如果任意一个桶的水量等于目标水量，就返回 true
                return True
            # 计算出所有可能的下一个状态
            nextStates = []
            # 把 1 桶灌满
            nextStates.append((jug1Capacity, curState[1]))
            # 把 2 桶灌满
            nextStates.append((curState[0], jug2Capacity))
            # 把 1 桶倒空
            nextStates.append((0, curState[1]))
            # 把 2 桶倒空
            nextStates.append((curState[0], 0))
            # 把 1 桶的水灌进 2 桶，直到 1 桶空了或者 2 桶满了
            nextStates.append((
                curState[0] - min(curState[0], jug2Capacity - curState[1]),
                curState[1] + min(curState[0], jug2Capacity - curState[1])
            ))
            # 把 2 桶的水灌进 1 桶，直到 2 桶空了或者 1 桶满了
            nextStates.append((
                curState[0] + min(curState[1], jug1Capacity - curState[0]),
                curState[1] - min(curState[1], jug1Capacity - curState[0])
            ))

            # 把所有可能的下一个状态都放进队列里
            for nextState in nextStates:
                # 把二维坐标转化为数字，方便去重
                hash = nextState[0] * (jug2Capacity + 1) + nextState[1]
                if hash in visited:
                    # 如果这个状态之前遍历过，就跳过，避免队列永远不空陷入死循环
                    continue
                q.append(nextState)
                visited.add(hash)
        return False
```