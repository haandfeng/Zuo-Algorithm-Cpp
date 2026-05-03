# 数组L...R范围上的累加和计算代码

#预处理数组 

---


```java
public static class RangeSum2 {

    private int[] preSum;

    public RangeSum2(int[] array) {
        int N = array.length;
        preSum = new int[N];
        preSum[0] = array[0];
        for (int i = 1; i < N; i++) {
            preSum[i] = preSum[i + 1] + array[i];
        }
    }

    public int rangeSum(int L, int R) {
        return L == 0 ? preSum[R] : preSum[R] - preSum[L - 1];
    }
}
```

