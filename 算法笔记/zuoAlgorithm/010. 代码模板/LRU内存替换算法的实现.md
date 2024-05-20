# LRU内存替换算法的实现

[M]

146.LRU 缓存机制
#LRU

---
https://leetcode-cn.com/problems/lru-cache

运用你所掌握的数据结构，设计和实现一个  LRU (最近最少使用) 缓存机制 。
实现 LRUCache 类：

LRUCache(int capacity) 以正整数作为容量 capacity 初始化 LRU 缓存
int get(int key) 如果关键字 key 存在于缓存中，则返回关键字的值，否则返回 -1 。
void put(int key, int value) 如果关键字已经存在，则变更其数据值；如果关键字不存在，则插入该组「关键字-值」。当缓存容量达到上限时，它应该在写入新数据之前删除最久未使用的数据值，从而为新的数据值留出空间。
 

进阶：你是否可以在 O(1) 时间复杂度内完成这两种操作？
```
示例：

输入
["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
[[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]
输出
[null, null, null, 1, null, -1, null, -1, 3, 4]

解释
LRUCache lRUCache = new LRUCache(2);
lRUCache.put(1, 1); // 缓存是 {1=1}
lRUCache.put(2, 2); // 缓存是 {1=1, 2=2}
lRUCache.get(1);    // 返回 1
lRUCache.put(3, 3); // 该操作会使得关键字 2 作废，缓存是 {1=1, 3=3}
lRUCache.get(2);    // 返回 -1 (未找到)
lRUCache.put(4, 4); // 该操作会使得关键字 1 作废，缓存是 {4=4, 3=3}
lRUCache.get(1);    // 返回 -1 (未找到)
lRUCache.get(3);    // 返回 3
```

提示：

1 <= capacity <= 3000
0 <= key <= 3000
0 <= value <= 10^4
最多调用 3 * 10^4 次 get 和 put

---
双向链表 + 哈希表


```java
public static class Node<K, V> {
    public K key;
    public V value;
    public Node<K, V> last;
    public Node<K, V> next;

    public Node(K key, V value) {
        this.key = key;
        this.value = value;
    }
}

// 双向链表
// 从head到tail所有节点都是串好的
public static class NodeDoubleLinkedList<K , V> {
    private Node<K, V> head;
    private Node<K, V> tail;

    public NodeDoubleLinkedList() {
        head = null;
        tail = null;
    }

    // 如果一个新的节点加入，放到尾巴上去
    public void addNode(Node<K, V> newNode) {
        if (newNode == null) {
            return;
        }
        // newNode != null
        if (head == null) { // 双向链表中一个节点也没有
            head = newNode;
            tail = newNode;
        } else { // 双向链表中之前有节点，tail（非null）
            tail.next = newNode;
            newNode.last = tail;
            tail = newNode;
        }
    }

    // 潜台词 ： 双向链表上一定有这个node
    // node分离出，但是node前后环境重新连接
    // node放到尾巴上去
    public void moveNodeToTail(Node<K, V> node) {
        if (this.tail == node) {
            return;
        }
        if (this.head == node) { // 当前node是老头部
            this.head = node.next;
            this.head.last = null;
        } else { // 当前node是中间的一个节点
            node.last.next = node.next;
            node.next.last = node.last;
        }
        node.last = this.tail;
        node.next = null;
        this.tail.next = node;
        this.tail = node;
    }

    public Node<K, V> removeHead() {
        if (this.head == null) {
            return null;
        }
        Node<K, V> res = this.head;
        if (this.head == this.tail) { // 链表中只有一个节点的时候
            this.head = null;
            this.tail = null;
        } else {
            this.head = res.next;
            res.next = null;
            this.head.last = null;
        }
        return res;
    }

}

public static class MyCache<K, V> {
    private HashMap<K, Node<K, V>> keyNodeMap;
    private NodeDoubleLinkedList<K, V> nodeList;
    private final int capacity;

    public MyCache(int cap) {
        if (cap < 1) {
            throw new RuntimeException("should be more than 0.");
        }
        keyNodeMap = new HashMap<K, Node<K, V>>();
        nodeList = new NodeDoubleLinkedList<K, V>();
        capacity = cap;
    }

    public V get(K key) {
        if (keyNodeMap.containsKey(key)) {
            Node<K, V> res = keyNodeMap.get(key);
            nodeList.moveNodeToTail(res);
            return res.value;
        }
        return null;
    }

    public void set(K key, V value) {
        if (keyNodeMap.containsKey(key)) {
            Node<K, V> node = keyNodeMap.get(key);
            node.value = value;
            nodeList.moveNodeToTail(node);
        } else { // 这是一个新加的记录，有可能出现替换
            Node<K, V> newNode = new Node<K, V>(key, value);
            keyNodeMap.put(key, newNode);
            nodeList.addNode(newNode);
            if (keyNodeMap.size() == capacity + 1) {
                removeMostUnusedCache();
            }
        }
    }

    private void removeMostUnusedCache() {
        Node<K, V> removeNode = nodeList.removeHead();
        keyNodeMap.remove(removeNode.key);
    }

}

```