# IndexTree代码

---


```java

// 下标从1开始！  
public static class IndexTree {  
  
   private int[] tree;  
   private int N;  
  
   // 0位置弃而不用！  
   public IndexTree(int size) {  
      N = size;  
      tree = new int[N + 1];  
   }  
  
   // 1~index 累加和是多少？  
   public int sum(int index) {  
      int ret = 0;  
      while (index > 0) {  
         ret += tree[index];  
         index -= index & -index;  
      }  
      return ret;  
   }  
  
   // index & -index : 提取出index最右侧的1出来  
   // index :           0011001000  
   // index & -index :  0000001000   public void add(int index, int d) {  
      while (index <= N) {  
         tree[index] += d;  
         index += index & -index;  
      }  
   }  
}

```