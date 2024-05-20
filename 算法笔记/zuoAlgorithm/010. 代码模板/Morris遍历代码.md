# Morris遍历代码

#Morris遍历 

---

```java

public static void morris(Node head) {
    if (head == null) {
        return;
    }
    Node cur = head;
    Node mostRight = null;
    while (cur != null) {
        // cur有没有左树
        mostRight = cur.left;
        if (mostRight != null) { // 有左树的情况下
            // 找到cur左树上，真实的最右
            while (mostRight.right != null && mostRight.right != cur) {
                mostRight = mostRight.right;
            }
            // 从while中出来，mostRight一定是cur左树上的最右节点
            // mostRight
            if (mostRight.right == null) {
                mostRight.right = cur;
                cur = cur.left;
                continue;
            } else { // mostRight.right != null -> mostRight.right == cur
                mostRight.right = null;
            }
        }
        cur = cur.right;
    }
}
    
```

- [[Morris实现前序遍历]]
- [[Morris实现中序遍历]]
- [[Morris实现后序遍历]]
