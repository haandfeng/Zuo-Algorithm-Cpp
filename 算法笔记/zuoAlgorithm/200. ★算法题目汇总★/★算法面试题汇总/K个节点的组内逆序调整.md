# K个节点的组内逆序调整

[H]

---

https://leetcode-cn.com/problems/reverse-nodes-in-k-group/

给定一个单链表的头节点head，和一个正数k
实现k个节点的小组内部逆序，如果最后一组不够k个就不调整
例子: 
调整前：1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8，k = 3
调整后：3 -> 2 -> 1 -> 6 -> 5 -> 4 -> 7 -> 8
``` java
public static ListNode reverseKGroup(ListNode head, int k) {  
   ListNode start = head;  
   ListNode end = teamEnd(start, k);  
   if (end == null) {  
      return head;  
   }  
   // 第一组很特殊因为牵扯到换头的问题  
   head = end;  
   reverse(start, end);  
   // 翻转之后start变成了上一组的结尾节点  
   ListNode lastTeamEnd = start;  
   while (lastTeamEnd.next != null) {  
      start = lastTeamEnd.next;  
      end = teamEnd(start, k);  
      if (end == null) {  
         return head;  
      }  
      reverse(start, end);  
      lastTeamEnd.next = end;  
      lastTeamEnd = start;  
   }  
   return head;  
}  
  
// 当前组的开始节点是s，往下数k个找到当前组的结束节点返回  
public static ListNode teamEnd(ListNode s, int k) {  
   while (--k != 0 && s != null) {  
      s = s.next;  
   }  
   return s;  
}  
  
// s -> a -> b -> c -> e -> 下一组的开始节点  
// 上面的链表通过如下的reverse方法调整成 : e -> c -> b -> a -> s -> 下一组的开始节点  
public static void reverse(ListNode s, ListNode e) {  
   e = e.next;  
   ListNode pre = null, cur = s, next = null;  
   while (cur != e) {  
      next = cur.next;  
      cur.next = pre;  
      pre = cur;  
      cur = next;  
   }  
   s.next = e;  
}
```