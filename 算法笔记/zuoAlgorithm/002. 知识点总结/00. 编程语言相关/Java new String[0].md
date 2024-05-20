Java # new String[0]

---

List 转数组时携带此参数防止转出类型出错

```java
List<String> ans = new ArrayList<>();
...

return ans.toArray(new String[0]);
```