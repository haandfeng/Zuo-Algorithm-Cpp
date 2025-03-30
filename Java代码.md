**一、HashSet 的创建与常规操作**

**✅ 创建 HashSet**

```java
import java.util.HashSet;
HashSet<String> set = new HashSet<>();
```

也可以指定初始容量或加载因子：
```java
HashSet<Integer> set = new HashSet<>(100); // 初始容量
```

**🔧 常规操作**
```java
set.add("apple");        // 添加元素
set.remove("apple");     // 删除元素
set.contains("apple");   // 判断是否包含某元素
set.size();              // 获取元素个数
set.isEmpty();           // 是否为空
set.clear();             // 清空集合
```

**🔁 遍历 HashSet**
```java
for (String item : set) {
    System.out.println(item);
}
```

**✅ 创建 HashMap**

```java
import java.util.HashMap;
HashMap<String, Integer> map = new HashMap<>();
```

也可以指定初始容量和负载因子：
```java
HashMap<String, Integer> map = new HashMap<>(128, 0.75f);
```
**🔧 常规操作**
```java
map.put("apple", 3);          // 添加或更新键值对
map.get("apple");             // 获取键对应的值（若不存在返回 null）
map.containsKey("apple");     // 是否包含某个键
map.containsValue(3);         // 是否包含某个值
map.remove("apple");          // 删除指定键的映射
map.size();                   // 键值对数量
map.isEmpty();                // 是否为空
map.clear();                  // 清空所有键值对
```

**🔁 遍历 HashMap**

1. 遍历键值对：
```java
for (Map.Entry<String, Integer> entry : map.entrySet()) {
    System.out.println(entry.getKey() + " -> " + entry.getValue());
}
```
2. 遍历键或值：
```java
for (Map.Entry<String, Integer> entry : map.entrySet()) {
    System.out.println(entry.getKey() + " -> " + entry.getValue());
}
```


**一、ArrayList（动态数组）**
**✅ 创建 ArrayList**

```java
import java.util.ArrayList;
ArrayList<String> list = new ArrayList<>();
```
也可以指定初始容量：
```java
ArrayList<Integer> list = new ArrayList<>(100);
```

```java
list.add("apple");         // 添加元素（默认尾部）
list.add(1, "banana");     // 在指定位置插入元素
list.get(0);               // 按索引获取元素
list.set(0, "orange");     // 替换指定索引的元素
list.remove(1);            // 删除指定位置的元素
list.remove("apple");      // 删除指定值的元素
list.contains("apple");    // 是否包含某元素
list.size();               // 元素个数
list.isEmpty();            // 是否为空
list.clear();              // 清空列表
```
**🔁 遍历方式**
```java
for (int i = 0; i < list.size(); i++) {
    System.out.println(list.get(i));
}

for (String item : list) {
    System.out.println(item);
}
```

**二、LinkedList（双向链表）**
**✅ 创建 LinkedList**

```java
import java.util.LinkedList;
LinkedList<String> list = new LinkedList<>();
```
**🔧 常规操作**
```java
list.add("apple");           // 添加元素（默认尾部）
list.addFirst("head");       // 在头部插入
list.addLast("tail");        // 在尾部插入
list.getFirst();             // 获取头部元素
list.getLast();              // 获取尾部元素
list.removeFirst();          // 移除头部元素
list.removeLast();           // 移除尾部元素
list.get(2);                 // 按索引获取元素
list.set(1, "banana");       // 替换指定位置的元素
list.contains("apple");      // 是否包含
list.size();                 // 元素数量
list.clear();                // 清空列表
```


**String操作**
**一、基本创建与拼接**
**✅ 创建字符串**
```java
String s1 = "hello";                  // 字面量创建
String s2 = new String("world");      // 构造方法创建（不推荐）

String result = s1 + s2;              // 使用 + 拼接（编译器优化为 StringBuilder）
result = s1.concat(s2);               // 使用 concat 方法
```

访问
```java
int len = s1.length();                // 字符串长度
char ch = s1.charAt(1);               // 获取索引为1的字符
s1.contains("el");                    // 是否包含子串
s1.indexOf("l");                      // 第一次出现的位置
s1.lastIndexOf("l");                  // 最后一次出现的位置
s1.startsWith("he");                 // 是否以...开头
s1.endsWith("lo");                   // 是否以...结尾

s1.equals(s2);                        // 内容是否相等（区分大小写）
s1.equalsIgnoreCase(s2);             // 忽略大小写比较
s1.compareTo(s2);                     // 字典序比较（返回正负或0）
```

截取和替换
```java
s1.substring(2);                      // 从索引2开始到结尾
s1.substring(1, 4);                   // 从索引1到3（不含4）
s1.substring(2);                      // 从索引2开始到结尾
s1.substring(1, 4);                   // 从索引1到3（不含4）

s1.trim();                            // 去除前后空格
s1.toUpperCase();                     // 转大写
s1.toLowerCase();                     // 转小写
```

切分和拼接
```java
String[] arr = s1.split(" ");         // 按空格分割（支持正则表达式）
String joined = String.join("-", arr);  // "a-b-c"

int n = Integer.parseInt("123");      // 字符串 -> int
String s = String.valueOf(123);       // 任意类型 -> 字符串
```


**栈**
```java
import java.util.Stack;

Stack<Integer> stack = new Stack<>();
stack.push(10);        // 入栈
stack.pop();           // 出栈（并返回栈顶元素）
stack.peek();          // 查看栈顶元素但不弹出
stack.isEmpty();       // 是否为空
stack.size();          // 栈中元素数量
```

**Deque**
```java
import java.util.Deque;
import java.util.ArrayDeque;
Deque<Integer> stack = new ArrayDeque<>();
stack.push(1);         // 入栈
stack.pop();           // 出栈
stack.peek();          // 查看栈顶
```

**二、队列 Queue**
Java 中的队列有多个实现，常用的是：
• LinkedList（经典）
• ArrayDeque（推荐）
• PriorityQueue（优先队列）

```java
import java.util.LinkedList;
import java.util.Queue;

Queue<String> queue = new LinkedList<>();
queue.offer("A");       // 入队（推荐，失败返回 false）
queue.add("B");         // 入队（失败抛异常）

queue.poll();           // 出队，返回队头并删除（空时返回 null）
queue.remove();         // 出队（空时抛异常）

queue.peek();           // 查看队头元素（不删除）
queue.isEmpty();        // 是否为空
queue.size();           // 元素数量
```


 **使用 ArrayDeque 实现队列（效率更好**
```java
import java.util.ArrayDeque;
Queue<Integer> queue = new ArrayDeque<>();
queue.offer(10);
queue.poll();
queue.peek();
```

**三、双端队列 Deque（双向操作）**

如果你想同时支持栈和队列的功能，用 Deque 非常合适：
```java
Deque<String> deque = new ArrayDeque<>();
deque.addFirst("a");   // 头部添加
deque.addLast("b");    // 尾部添加
deque.removeFirst();   // 从头部移除
deque.removeLast();    // 从尾部移除
```

**四、优先队列（PriorityQueue）**

```java
import java.util.PriorityQueue;
PriorityQueue<Integer> pq = new PriorityQueue<>();
pq.offer(3);
pq.offer(1);
pq.offer(2);

System.out.println(pq.poll());   // 输出：1（默认小顶堆）
```

你也可以用 PriorityQueue<>(Comparator.reverseOrder()) 实现大顶堆。