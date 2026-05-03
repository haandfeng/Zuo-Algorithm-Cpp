# a~b这些数字随意选m个不降序的结果数

#排列组合

---

a~b这些数字随意选m个不降序的结果数:  

$C_{b-a+m}^m$

[[组合数代码]]

```java
// 从一共a个数里，选b个数，方法数是多少
public static long c(int a, int b) {
    if (a == b) {
        return 1;
    }
    long x = 1;
    long y = 1;
    for (int i = b + 1, j = 1; i <= a; i++, j++) {
        x *= i;
        y *= j;
        long gcd = gcd(x, y);
        x /= gcd;
        y /= gcd;
    }
    return x / y;
}
```

ref: [[排列组合]]