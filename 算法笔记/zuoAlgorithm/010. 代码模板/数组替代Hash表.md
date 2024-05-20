# 数组替代Hash表

---


#### 数组做词频统计

```java
char[] str1 = s1.toCharArray();
int[] count = new int[26];
for (char cha : str1) {
    count[cha - 'a']++;
}
```
[[多少张贴纸可以贴出给定字符串|贴纸拼词]]  