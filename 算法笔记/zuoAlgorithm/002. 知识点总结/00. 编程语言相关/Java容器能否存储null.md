# Java容器能否存储null

---

1、Arraylist、Linkedlist可以存储多个null.

2、Hashset、LinkedHashset可以存储一个null. Treeset不能存储null.

3、Hashmap 、LinkedHashmap  key与value均可以为null.  Treemap  key不可以为null,value可以为null.  hashtable、concurrenthashmap key与value均不能为null.


```java
public static void main(String[] args) {  
    List<Integer> list = new ArrayList<>();  
    list.add(null);  
    list.add(null);  
    System.out.println("ArrayList :" + list.size());   //2  
  
    List<Integer> linkedList = new LinkedList<>();  
    linkedList.add(null);  
    linkedList.add(null);  
    System.out.println("LinkedList :" + linkedList.size());  //2  
  
    Set<Integer> set = new HashSet<>();  
    set.add(null);  
    set.add(null);  
    System.out.println("HashSet :" + set.size());// 1  
  
    Set<Integer> linkedHashSet = new LinkedHashSet<>();  
    linkedHashSet.add(null);  
    linkedHashSet.add(null);  
    System.out.println("LinkedHashSet :" + linkedHashSet.size()); //1  
  
    TreeSet<String> treeSet = new TreeSet<>();  
    //treeSet.add(null);   //treeSet key不能null ava.lang.NullPointerException  
    //treeSet.add(null);    System.out.println("TreeSet : " + treeSet.size());  
  
    HashMap<String, String> map = new HashMap<>();  
    map.put(null, null);  
    map.put(null, "aaa"); // 相同key, value被覆盖  
    map.put("aaa", null);  
    System.out.println("HashMap :" + map.size());// 2  
  
    LinkedHashMap<String, String> linkedHashMap = new LinkedHashMap<>();  
    linkedHashMap.put(null, null);  
    linkedHashMap.put("aaa", null);  
    System.out.println("LinkedHashMap :" + linkedHashMap.size()); //2  
  
    TreeMap<String, String> treeMap = new TreeMap<>();  
    treeMap.put("aaa", null);   //treeMap key  不能为空，value 可以为空  
    //treeMap.put(null,"aaa");  
    //treeMap.put(null,null);    System.out.println("TreeMap :" + treeMap.size());  
  
    Hashtable<String, String> hashtable = new Hashtable<>();  
    // hashtable.put("aaa",null); // hashtable key与value均不能为空  
    // hashtable.put(null,"aaa");  
    // hashtable.put(null, null);    System.out.println("Hashtable" + hashtable.size());  
  
    ConcurrentHashMap<String, String> concurrentHashMap = new ConcurrentHashMap<>();  
    concurrentHashMap.put("aaa", null);    // concurrentHashmap key 与 value 均不可以为空  
    //concurrentHashMap.put(null,"aaa");  
    System.out.println("ConcurrentHashMap :" + concurrentHashMap.size());  
}

```