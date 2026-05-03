# 返回离非负整数num最近的2的某次方

---
给定一个非负整数num，如何不用循环语句，返回>=num，并且离num最近的，2的某次方


```java
// 已知n是正数
// 返回大于等于，且最接近n的，2的某次方的值
public static final int tableSizeFor(int n) {
    n--;
    n |= n >>> 1;
    n |= n >>> 2;
    n |= n >>> 4;
    n |= n >>> 8;
    n |= n >>> 16;
    return (n < 0) ? 1 : n + 1;
}
```