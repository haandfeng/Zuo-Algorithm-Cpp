# 枚举所有的i行到j行的情况


---

- [[给定一个整型矩阵，返回子矩阵的最大累加和]]

```java
public static int maxSum(int[][] m) {  
   if (m == null || m.length == 0 || m[0].length == 0) {  
      return 0;  
   }  
   int max = Integer.MIN_VALUE;  
   int cur = 0;  
   int[] s = null;  
   for (int i = 0; i != m.length; i++) { // 开始的行号i  
 s = new int[m[0].length]; //   
for (int j = i; j != m.length; j++) { // 结束的行号j，i~j行是我讨论的范围  
 cur = 0;  
         for (int k = 0; k != s.length; k++) {  
            s[k] += m[j][k];  
            cur += s[k];  
            max = Math.max(max, cur);  
            cur = cur < 0 ? 0 : cur;  
         }  
      }  
   }  
   return max;  
}


```



- [[返回数组中，有多少个独立的域]]

```java
public static int largestComponentSize(int[] arr) {
    UnionFindSet1 set = new UnionFindSet1(arr.length);
    for (int i = 0; i < arr.length; i++) {
        for (int j = i + 1; j < arr.length; j++) {
            if (gcd(arr[i], arr[j]) != 1) {
                set.union(i, j);
            }
        }
    }
    return set.maxSize();
}
```