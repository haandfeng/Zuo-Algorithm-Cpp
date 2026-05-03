# 深度复制带有rand指针的链表

#链表 

[M]

---
https://leetcode-cn.com/problems/copy-list-with-random-pointer



一种特殊的单链表节点类描述如下 
```java
class Node { 
	int value; 
	Node next; 
	Node rand; 
	Node(int val) { value = val; } 
} 
```
rand指针是单链表节点结构中新增的指针，rand可能指向链表中的任意一个节点，也可能指向null。 
给定一个由Node节点类型组成的无环单链表的头节点 head，请实现一个函数完成这个链表的复制，并返回复制的新链表的头节点。   
【要求】  
时间复杂度O(N)，额外空间复杂度O(1)   
