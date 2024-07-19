# 序列化和反序列化 N 叉树

428.序列化和反序列化 N 叉树

[H]

---

n叉树的 https://leetcode-cn.com/problems/serialize-and-deserialize-n-ary-tree/
二叉树的 https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/

序列化是指将一个数据结构转化为位序列的过程，因此可以将其存储在文件中或内存缓冲区中，以便稍后在相同或不同的计算机环境中恢复结构。 

设计一个序列化和反序列化 N 叉树的算法。一个 N 叉树是指每个节点都有不超过 N 个孩子节点的有根树。序列化 / 反序列化算法的算法实现没有限制。你只需要保证 N 叉树可以被序列化为一个字符串并且该字符串可以被反序列化成原树结构即可。  

例如，你需要序列化下面的 3-叉 树。  
![[Pasted image 20211106205247.png|300]]  
  
```
为 [1 [3[5 6] 2 4]]。你不需要以这种形式完成，你可以自己创造和实现不同的方法。

或者，您可以遵循 LeetCode 的层序遍历序列化格式，其中每组孩子节点由空值分隔。
```
![[Pasted image 20211106205313.png|400]]  
```
例如，上面的树可以序列化为 [1,null,2,3,4,5,null,null,6,7,null,8,null,9,10,null,null,11,null,12,null,13,null,null,14]

你不一定要遵循以上建议的格式，有很多不同的格式，所以请发挥创造力，想出不同的方法来完成本题。

提示：
树中节点数目的范围是 [0, 104].
0 <= Node.val <= 104
N 叉树的高度小于等于 1000
不要使用类成员 / 全局变量 / 静态变量来存储状态。你的序列化和反序列化算法应是无状态的。
```

## 先序
```java
// 二叉树先序序列化和反序列化  
// 测试链接 : https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/public class Code05_PreorderSerializeAndDeserialize {  
  
   // 不提交这个类  
   public static class TreeNode {  
      public int val;  
      public TreeNode left;  
      public TreeNode right;  
  
      public TreeNode(int v) {  
         val = v;  
      }  
   }  
  
    // 二叉树可以通过先序、后序或者按层遍历的方式序列化和反序列化  
    // 但是，二叉树无法通过中序遍历的方式实现序列化和反序列化  
    // 因为不同的两棵树，可能得到同样的中序序列，即便补了空位置也可能一样。  
    // 比如如下两棵树  
    //         __2  
    //        /    //       1    //       和  
    //       1__  
    //          \    //           2    // 补足空位置的中序遍历结果都是{ null, 1, null, 2, null}  
   // 提交这个类  
   public class Codec {  
  
      public String serialize(TreeNode root) {  
         StringBuilder builder = new StringBuilder();  
         f(root, builder);  
         return builder.toString();  
      }  
  
      void f(TreeNode root, StringBuilder builder) {  
         if (root == null) {  
            builder.append("#,");  
         } else {  
            builder.append(root.val + ",");  
            f(root.left, builder);  
            f(root.right, builder);  
         }  
      }  
  
      public TreeNode deserialize(String data) {  
         String[] vals = data.split(",");  
         cnt = 0;  
         return g(vals);  
      }  
  
      // 当前数组消费到哪了  
      public static int cnt;  
  
      TreeNode g(String[] vals) {  
         String cur = vals[cnt++];  
         if (cur.equals("#")) {  
            return null;  
         } else {  
            TreeNode head = new TreeNode(Integer.valueOf(cur));  
            head.left = g(vals);  
            head.right = g(vals);  
            return head;  
         }  
      }  
  
   }  
  
}
```

## 层序
```java
// 二叉树按层序列化和反序列化  
// 测试链接 : https://leetcode.cn/problems/serialize-and-deserialize-binary-tree/public class Code06_LevelorderSerializeAndDeserialize {  
  
   // 不提交这个类  
   public static class TreeNode {  
      public int val;  
      public TreeNode left;  
      public TreeNode right;  
  
      public TreeNode(int v) {  
         val = v;  
      }  
   }  
  
   // 提交这个类  
   // 按层序列化  
   public class Codec {  
  
      public static int MAXN = 10001;  
  
      public static TreeNode[] queue = new TreeNode[MAXN];  
  
      public static int l, r;  
  
      public String serialize(TreeNode root) {  
         StringBuilder builder = new StringBuilder();  
         if (root != null) {  
            builder.append(root.val + ",");  
            l = 0;  
            r = 0;  
            queue[r++] = root;  
            while (l < r) {  
               root = queue[l++];  
               if (root.left != null) {  
                  builder.append(root.left.val + ",");  
                  queue[r++] = root.left;  
               } else {  
                  builder.append("#,");  
               }  
               if (root.right != null) {  
                  builder.append(root.right.val + ",");  
                  queue[r++] = root.right;  
               } else {  
                  builder.append("#,");  
               }  
            }  
         }  
         return builder.toString();  
      }  
  
      public TreeNode deserialize(String data) {  
         if (data.equals("")) {  
            return null;  
         }  
         String[] nodes = data.split(",");  
         int index = 0;  
         TreeNode root = generate(nodes[index++]);  
         l = 0;  
         r = 0;  
         queue[r++] = root;  
         while (l < r) {  
            TreeNode cur = queue[l++];  
            cur.left = generate(nodes[index++]);  
            cur.right = generate(nodes[index++]);  
            if (cur.left != null) {  
               queue[r++] = cur.left;  
            }  
            if (cur.right != null) {  
               queue[r++] = cur.right;  
            }  
         }  
         return root;  
      }  
  
      private TreeNode generate(String val) {  
         return val.equals("#") ? null : new TreeNode(Integer.valueOf(val));  
      }  
  
   }  
  
}
```