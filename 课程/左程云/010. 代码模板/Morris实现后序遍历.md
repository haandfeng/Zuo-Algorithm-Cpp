# Morris实现后序遍历

#Morris遍历 

---

```java
public static void morrisPos(Node head) {
    if (head == null) {
        return;
    }
    Node cur = head;
    Node mostRight = null;
    while (cur != null) {
        mostRight = cur.left;
        if (mostRight != null) {
            while (mostRight.right != null && mostRight.right != cur) {
                mostRight = mostRight.right;
            }
            if (mostRight.right == null) {
                mostRight.right = cur;
                cur = cur.left;
                continue;
            } else {
                mostRight.right = null;
                printEdge(cur.left);
            }
        }
        cur = cur.right;
    }
    printEdge(head);
    System.out.println();
}

public static void printEdge(Node head) {
    Node tail = reverseEdge(head);
    Node cur = tail;
    while (cur != null) {
        System.out.print(cur.value + " ");
        cur = cur.right;
    }
    reverseEdge(tail);
}

public static Node reverseEdge(Node from) {
    Node pre = null;
    Node next = null;
    while (from != null) {
        next = from.right;
        from.right = pre;
        pre = from;
        from = next;
    }
    return pre;
}

```