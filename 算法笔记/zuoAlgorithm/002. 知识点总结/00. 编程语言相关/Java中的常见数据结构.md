# Java中的常见数据结构


---

### 双向链表/双端队列
```java
LinkedList<Integer> qmax = new LinkedList<Integer>();

qmax.peekLast()  // 尾部值
qmax.peekFirst() // 头部值

qmax.pollLast()  // 弹出尾部
qmax.pollFirst() // 弹出头部

qmax.isEmpty()   // 判空
```


### 优先队列

#### 小根堆
```java
// 小顶堆
PriorityQueue<Integer> pQ = new PriorityQueue<>();

pQ.add(arr[i])  // 加入
pQ.poll()       // 弹出
```


#### 大根堆
需要定义比较器
```java
//  负数，o1 放在上面的情况
public static class MyComp implements Comparator<Integer>{
    @Override
    public int compare(Integer o1, Integer o2) {
        return o2 - o1;
    }
}

PriorityQueue<Integer> heap = new PriorityQueue<>(new MyComp());

heap.add(5);
heap.add(7);
heap.add(3);
heap.add(0);
heap.add(2);
heap.add(5);

while(!heap.isEmpty()) {
    System.out.println(heap.poll());
}

```


