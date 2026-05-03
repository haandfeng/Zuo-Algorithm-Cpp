# 一种高效的insert跟get结构

算法课练习小题


---

ArrayList 扩容, 均摊代价 O(1)
在index 位置插入一个 O(n)
>ArrayList 是一个可以扩容的东西,  
一开始有一个初始长度, 往里加数, 加满了, 它会扩容  
大小翻一倍  
问: 即便算上扩容代价, 这个扩容代价均摊下来是 O(1)  
底层就是一个连续的数组结构  
ArrayList中在index 位置插入一个number 代价是 O(n)  
需要把后面所有东西整体向右挪一个位置, 腾出这个位置,   
然后这个number进来, 这个是O(N)的

LinkedList: 双向链表
index位置插入一个number需要做遍历
>LinkedList: 双向链表  
在index位置插入一个number, 代价是O(1)吗?  
不是,   
你如果找到number应该插在哪儿, 它是O(1),   
问题是你怎么找到index在哪儿? 得for循环  
从0开始, 数够index个, 才能把number插进来  
所以它也是O(N)的

两个方法在index插入一个数, 都是O(N)的  
一个是后面的东西要往右移,   
一个是找到index位置在哪儿我才能插  
能不能设计一个结构, 在index 位置插入number  
比O(N)的代价低, get index位置的值也比O(N)的代价低  
这个结构改如何设计?


提示:   
用平衡搜索二叉树改的  
最后insert 跟 get 都是 O(logN)级别

---

讲解@leetcode高频题目全讲(12)
