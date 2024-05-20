# Java中HashSet vs TreeSet

---

**HashSet**
其底层数据结构是哈希表，不保证顺序。
在插入对象类型的时候，默认可以重复（**属性值完全相同的对象**），只有重写实体类的hashcode()和equals()方法后才可以防止插入重复对象。
a. 首先通过hashcode()判断要插入对象的hashcode是否与set中已有对象相等，如果没有相等的则为不同的对象可以插入set，
b. 如果有相等则继续通过equals()比较两个对象的每个属性是否完全相等，如相等则为重复对象，不能插入。
**hashset在插入基本数据类型的时候不能插入重复值**。

**TreeSet**
底层是TreeMap(红黑树)，可以给Set集合中的元素进行指定顺序的排序。

自然排序：
a. 让实现类实现接口：Comparable, 泛型：要排序的实体类.
b. 重写方法，compareTo() .
**0： 代表相同的元素， 不添加到TreeSet中；**
非0： 则需要添加到集合；return -1； 从小到大排列。

定制排序：
a. 定义一个类，实现Comparator接口， --自定义比较器.
b. 重写方法，compare().
c. 创建一个TreeSet对象时，把自定义的比较器，放入到TreeSet的比较器中。

三. HashSet vs TreeSet

**不同之处**
像add,remove,contains,size等操作，HashSet比TreeSet有更好的性能。
HashSet的时间复杂度为O(1)，TreeSet的时间复杂度为log(n)。
HashSet是无序的，TreeSet是有序的，默认为增序。

**相同之处**
都没有重复的元素


如果你想要一个有序集合，建议先把元素添加到HashSet中，然后把它转换为TreeSet。
 `Set<Integer> treeSet = new TreeSet<Integer>(hashset);`
 
它们都是非线程安全的。

HashSet内部主要使用HashMap实现，HashMap不允许有相同的key存在。
TreeSet内部主要依靠TreeMap来实现。
