# Java中的哈希表/散列表

---


```text
HashMap 是一个散列表，它存储的内容是键值对(key-value)映射。  
1.通过keySet()获取键，再利用hashmap里面的.get(key)方法通过键获取值   
2.通过Map.entry()获取键值对，可以同时利用迭代器直接遍历   
3.通过父类Collection获取值   
Hashtable  
数组+链表实现，key，value都不能为null，线程安全，实现方式是修改数据时锁住整个
HashTable，效率低。
初始大小11，扩容\*2+1。
HashMap  
数组+链表实现，key，value都可以为null，线程不安全。
初始大小16，扩容\*2。

ConcurrentHashMap  
分段数组+链表实现，线程安全。用分离锁实现，允许多个修改操作并发。
跨段修改时按顺序锁定所有段，再按顺序释放。
```


Java中的哈希表分为 (内部)按值传递的哈希表 和 按 (内部)引用传递的哈希表, 区别很大

String 在Java的其它地方有可能按引用传递, 但在哈希表内部是按值传递的

Hashmap的 put 既是新增操作, 也是更新value的操作

哈希表不管存了多少数据, 增删改查都是常数时间, 但是这个常数时间是比较大的
它远远比数组寻址, 加减法等常数时间的操作要大的多


哈希表内部 Integer, Double, Float, String, Char全部按值传递
```java
String test1 = "aaaa";
String test2 = "aaaa";
System.out.println(map.containsKey(test1));
System.out.println(map.containsKey(test2));

HashMap<Integer, String> map2 = new HashMap<>();
map2.put(1234567, "我是1234567");

Integer a = 1234567;
Integer b = 1234567;

System.out.println(a == b);
System.out.println(map2.containsKey(a));
System.out.println(map2.containsKey(b));

```

而非原生类型, 比如自己定义的, 内部按引用传递, 内存地址作为key
```java
public static class Node {
    public int value;

    public Node(int v) {
        value = v;
    }
}

Node node1 = new Node(1);
Node node2 = new Node(1);
HashMap<Node, String> map3 = new HashMap<>();
map3.put(node1, "我进来了！");
System.out.println(map3.containsKey(node1));
System.out.println(map3.containsKey(node2));

```




