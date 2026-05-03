
# 树节点的第 K 个祖先

#倍增法

1483.树节点的第 K 个祖先


---

给你一棵树，树上有 n 个节点，按从 0 到 n-1 编号。树以父节点数组的形式给出，其中 parent[i] 是节点 i 的父节点。树的根节点是编号为 0 的节点。

树节点的第 k 个祖先节点是从该节点到根节点路径上的第 k 个节点。

实现 TreeAncestor 类：

TreeAncestor（int n， int[] parent） 对树和父数组中的节点数初始化对象。
getKthAncestor(int node, int k) 返回节点 node 的第 k 个祖先节点。如果不存在这样的祖先节点，返回 -1 。
![[Pasted image 20220421140849.png]]

```
输入：
["TreeAncestor","getKthAncestor","getKthAncestor","getKthAncestor"]
[[7,[-1,0,0,1,1,2,2]],[3,1],[5,2],[6,3]]

输出：
[null,1,0,-1]

解释：
TreeAncestor treeAncestor = new TreeAncestor(7, [-1, 0, 0, 1, 1, 2, 2]);

treeAncestor.getKthAncestor(3, 1);  // 返回 1 ，它是 3 的父节点
treeAncestor.getKthAncestor(5, 2);  // 返回 0 ，它是 5 的祖父节点
treeAncestor.getKthAncestor(6, 3);  // 返回 -1 因为不存在满足要求的祖先节点
 

提示：

1 <= k <= n <= 5 * 10^4
parent[0] == -1 表示编号为 0 的节点是根节点。
对于所有的 0 < i < n ，0 <= parent[i] < n 总成立
0 <= node < n
至多查询 5 * 10^4 次
```