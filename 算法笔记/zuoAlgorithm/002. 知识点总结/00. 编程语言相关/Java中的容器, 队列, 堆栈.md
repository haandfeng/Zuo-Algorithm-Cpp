# Java中的容器, 队列, 堆,  栈

---



ArrayList: 用作数组

## 队列 Queue
没有叫做Queue的类（它是个接口名字）
Queue只是一个接口，当需要使用队列时也就首选ArrayDeque了（次选是LinkedList）

LinkedList: 双向链表-->双端队列

```java
LinkedList<Node> queue = new LinkedList<>();

```

## 栈: Stack
当需要使用栈时，Java已不推荐使用Stack，而是推荐使用更高效的ArrayDeque

```java
Stack<Integer> stack = new Stack<>();


Deque<Integer> stack = new ArrayDeque<Integer>();
```


[[Java中的Stack结构]]



## 堆

优先队列 - PriorityQueue
如果不提供Comparator的话，优先队列中元素默认按自然顺序排列，也就是数字默认是小的在队列头，字符串则按字典序排列。
如果需要大顶堆, 需要定义比较器

```java
// double类型如何定义大根堆
PriorityQueue<int[]> pq = new PriorityQueue<int[]>((a, b) ->  
        Double.compare((double) b[0] / b[1], (double) a[0] / a[1]));

```

## String 转为 char

使用 String.toCharArray( ) 方法，将String 转化为 字符串数组。(返回值：char[] ) 

char 转为 String:  
String s = String.valueOf('c');







