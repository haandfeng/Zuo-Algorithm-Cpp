# Java中的Treeset和Priorityqueue

---
## **应用场景**

**Treeset**
需要找出集合中大于等于某个数值的最小值（ceiling()方法），或小于等于某个数值的最大值（floor()方法），即有不等关系的需求（时间复杂度为二分法的时间复杂度）（还有higher()和lower()方法分别代表大于以及小于某个值）

**Priorityqueue**
需要找出集合中的最大值或最小值时，即有最值的需求



**相同：**

删除头节点的时间复杂度是O(logn)
添加节点的时间复杂度是O(logn)
都不是线程安全

**区别：**

treeset是红黑树（介于二叉搜索树和平衡二叉搜索树之间，树的深度即不会过深，也不会在调整平衡时，花费过多的时间）  
Priorityqueue是完全二叉树（除了叶节点不满，其他层全是满的）

删除某一个指定的元素，Priorityqueue是O(n)，Treeset是O(logn)，
因为Priorityqueue的查找节点时间是O(n)，treeset是O(logn)

treeset的iterator是有序遍历，但是Priorityqueue不是