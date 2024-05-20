# 一个简单的Hash函数

---

假设字符都是小写字母
除余法实现:

```java
class HashFunction {
    private int cap;
    private int seed;
    public HashFunction(int cap, int seed) {
        this.cap = cap;
        this.seed = seed;
    }

    int myHash(String str) {
        int hashcode = 0;
        int n = str.length();
        for (int i = 0; i < n; i++) {
            hashcode += seed * hashcode + (str.charAt(i) - 'a');
            hashcode %= cap;
        }
        return hashcode;
    }
};
```