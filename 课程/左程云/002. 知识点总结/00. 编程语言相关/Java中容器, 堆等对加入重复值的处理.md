# Java中容器, 堆等对加入重复值的处理

---


## 堆里接受加入重复值
比如: Java里的 PriorityQueue


## 有序表不允许加入重复值
比如: Java里的 TreeMap

有序表如果想加入重复值, 需要在相等时把地址传入比较器
对于简单数据结果, 可以再包一层

ref: [[Java Object 之hashcode]]

```java
public static void main(String[] args) {  
    int[] nums1 = new int[2];  
    int[] nums2 = new int[2];  
    System.out.println(nums1.hashCode());  
    System.out.println(nums2.hashCode());  
  
    System.out.println(nums1.hashCode());  
    System.out.println(nums2.hashCode());  
}
```


```java
private TreeSet<int[]> treeSet  
        = new TreeSet<>((o1, o2) -> o1[0] == o2[0] ?  
        // (o1.hashCode() - o2.hashCode()) :  
        (o1.toString().compareTo(o2.toString())) :  
        o1[0] - o2[0]);

```

ref: 2034. [[股票价格波动]]


