# 调整数出现的概率从x调整为x^2

---


如果利用Math.random()函数，
把得到\[0,x)范围上的数的概率从x调整成x^2


```java
// 返回[0,1)的一个小数
// 任意的x，x属于[0,1)，[0,x)范围上的数出现概率由原来的x调整成x平方
public static double xToXPower2() {
    return Math.max(Math.random(), Math.random());
}

count = 0;
double x = 0.17;
for (int i = 0; i < testTimes; i++) {
    if (xToXPower2() < x) {
        count++;
    }
}
System.out.println((double) count / (double) testTimes);
System.out.println((double) 1 - Math.pow((double) 1 - x, 2));

```

把得到\[0,x)范围上的数的概率从x调整成x^3
```java
// 返回[0,1)的一个小数
// 任意的x，x属于[0,1)，[0,x)范围上的数出现概率由原来的x调整成x立方
public static double xToXPower3() {
    return Math.max(Math.random(), Math.max(Math.random(), Math.random());
}

count = 0;
double x = 0.17;
for (int i = 0; i < testTimes; i++) {
    if (xToXPower3() < x) {
        count++;
    }
}
System.out.println((double) count / (double) testTimes);
System.out.println((double) 1 - Math.pow((double) 1 - x, 2));



```