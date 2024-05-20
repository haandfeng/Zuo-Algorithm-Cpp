# Morris遍历
---

## [[Morris遍历]] 


一种遍历二叉树的方式，并且时间复杂度O(N)，额外空间复杂度O(1)  
 
通过利用原树中大量空闲指针的方式，达到节省空间的目的  


## Morris遍历细节

假设来到当前节点cur，开始时cur来到头节点位置  
1）如果cur没有左孩子，cur向右移动(cur = cur.right)  
2）如果cur有左孩子，找到左子树上最右的节点mostRight：  
	a. 如果mostRight的右指针指向空，让其指向cur，然后cur向左移动(cur = cur.left)  
	b. 如果mostRight的右指针指向cur，让其指向null，然后cur向右移动(cur = cur.right)  
3）cur为空时遍历停止  

- [[Morris遍历代码]]
- [[Morris实现前序遍历]]
- [[Morris实现中序遍历]]
- [[Morris实现后序遍历]] : 比较难

## Morris遍历实质

建立一种机制：

>对于没有左子树的节点只到达一次，  
对于有左子树的节点会到达两次   
morris遍历时间复杂度依然是O(N)  

- [[二叉树的最小深度]]  Morris遍历实现  
- [[判断二叉树是不是平衡二叉树]]  Morris遍历实现  