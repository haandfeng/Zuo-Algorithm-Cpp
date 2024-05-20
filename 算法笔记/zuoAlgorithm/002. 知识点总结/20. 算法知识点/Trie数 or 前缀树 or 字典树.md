# Trie数 or 前缀树 or 字典树

#前缀树 
#Trie


---
- [[字母表实现前缀树]]
- [[哈希表实现前缀树]]


## 题目

## 经典前缀树


### 改写的前缀树

前缀树上每条路放的是字符串, 不是经典前缀树中的字符
- [[打印目录结构]]

```java
public static class Node {  
   // 上一个节点是通过哪条路，到我的  
 public String path;  
   // key : node下级的路 value：node在key这条路上对应的节点是什么  
 public TreeMap<String, Node> nextMap;  
     
   public Node(String p) {  
      this.path = p;  
      nextMap = new TreeMap<>();  
   }  
}


```

