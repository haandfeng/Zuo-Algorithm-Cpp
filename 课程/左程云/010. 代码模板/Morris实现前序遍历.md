# Morris实现前序遍历

#Morris遍历 

---


```java
public static void morrisPre(Node head) {
    if (head == null) {
        return;
    }
    Node cur1 = head;
    Node cur2 = null;
    while (cur1 != null) {
        cur2 = cur1.left;
        if (cur2 != null) {
            while (cur2.right != null && cur2.right != cur1) {
                cur2 = cur2.right;
            }
            if (cur2.right == null) {
                cur2.right = cur1;
                System.out.print(cur1.value + " ");
                cur1 = cur1.left;
                continue;
            } else {
                cur2.right = null;
            }
        } else {
            System.out.print(cur1.value + " ");
        }
        cur1 = cur1.right;
    }
    System.out.println();
}

```