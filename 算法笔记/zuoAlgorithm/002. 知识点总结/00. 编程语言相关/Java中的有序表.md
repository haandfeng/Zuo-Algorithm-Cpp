# Java中的有序表 
---

TreeMap  
TreeSet  

有序表是一种规范, 它是接口名  
 key: 有序组织  
 增删改查: O(logN)  
任何一个数据结构满足这些, 都是有序表  
 AVL, SB树, 红黑树 都可以实现有序表, 性能指标几乎无差别  
 跳表也是  
 

java中有序表是红黑树   
有序表可以被：红黑树、avl树、跳表、size-balanced-tree(SB树)实现  
不同的实现有什么区别：在使用层次上和性能上看，没区别。只有常数时间的区别。  
所有接口的性能O(logN)  
设计细节：扩展班最后一节  


有序表和哈希表的区别：  
哈希表的所有功能有序表一定有，哈希表的key（散乱组织，哈希函数） 
有序表所有的key有序组织，比哈希表的功能多。 
哈希表所有操作，使用时认为时间复杂度O(1)，有序表所有接口的性能O(logN)   


```java
TreeMap<Integer, String> map = new TreeMap<>();
map.put(7, "我是7"); // （key,value） 所有的key按顺序组织
map.put(3, "我是3");
map.put(9, "我是9");
map.put(2, "我是2");
map.put(8, "我是8");
map.put(5, "我是5");

map.put(5, "我还是5");// (5, 我还是5)


System.out.println(map.containsKey(2));
System.out.println(map.get(7));
map.remove(9);




System.out.println(map.firstKey());
System.out.println(map.lastKey());
// <= num 离num最近的key
System.out.println(map.floorKey(6));
System.out.println(map.floorKey(-2));
// >= num 离num最近的东西
System.out.println(map.ceilingKey(6));
System.out.println(map.ceilingKey(100));

// 时间复杂度全是O(logN)级别


```

如果是Node类型, 有序表不能直接用, 因为要求key一定是要求可以比较的东西, 需要在构造参数里传递比较器

```java
Node node3 = new Node(3);
Node node4 = new Node(4);
TreeMap<Node, String> treeMap2 = new TreeMap<>();
treeMap2.put(node3, "我是node3");
treeMap2.put(node4, "我是node4");
```