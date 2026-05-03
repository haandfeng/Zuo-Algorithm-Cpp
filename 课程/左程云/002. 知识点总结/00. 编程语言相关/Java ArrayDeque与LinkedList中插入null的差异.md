# Java ArrayDeque与LinkedList中插入null的差异

---

如果要对Deque进行加入null元素的操作，选择LinkedList类


用ArrayDeque类插入null时会报NullPointerException的错误，而用LinkedList类则不会。

ArrayDeque类里是对offer这个方法特意进行了是否为null的判断
![[Pasted image 20220827085505.png]]

而LinkedList类则没有进行这个判断
![[Pasted image 20220827085451.png]]
