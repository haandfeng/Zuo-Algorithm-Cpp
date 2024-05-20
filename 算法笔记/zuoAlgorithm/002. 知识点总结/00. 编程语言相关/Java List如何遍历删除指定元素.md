# Java List如何遍历删除指定元素

---


ref: https://blog.csdn.net/LosingCarryJie/article/details/86693813

正向删除可能会导致数组下标挪动导致错位  
**逆序遍历解决问题**

```java
for (int i = list.size()-1; i >=0; i--) {
    if(list.get(i)%2==0){
        list.remove(i);
    }
}
```