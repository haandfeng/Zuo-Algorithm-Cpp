# Java Double类型取模及精度控制

---



ref: 774.[[最小化去加油站的最大距离]] [H]

```java

public static boolean ok1(double limit, int[] stations, int K) {
    int used = 0;
    for (int i = 1; i < stations.length; i++) {
        double gap = stations[i] - stations[i - 1];
        if(gap - limit > 1e-6){ // 大于1个limit的才需要塞入
            double cnt = gap / limit;
            double modCnt = gap % limit; // IMP: 学习doube取模
            if (modCnt <= 1e-6) {
                used += (int) cnt - 1;
            } else {
                used += (int) cnt;
            }
        }
        if (used > K) {
            return false;
        }
    }
    return true;
}

```

