# DP二维表只遍历右上半部区



----



- [[整个人物关系中有多少个朋友圈]]

```java
public static int findCircleNum(int[][] M) {
    int N = M.length;
    // {0} {1} {2} {N-1}
    UnionFind unionFind = new UnionFind(N);
    for (int i = 0; i < N; i++) {
        for (int j = i + 1; j < N; j++) {
            if (M[i][j] == 1) { // i和j互相认识
                unionFind.union(i, j);
            }
        }
    }
    return unionFind.sets();
}
```

- [[最少添加多少字符使字符串变为回文串]]

```java
	public static int[][] getDP(char[] str) {
		int[][] dp = new int[str.length][str.length];
		for (int j = 1; j < str.length; j++) {
			dp[j - 1][j] = str[j - 1] == str[j] ? 0 : 1;
			for (int i = j - 2; i > -1; i--) {
				if (str[i] == str[j]) {
					dp[i][j] = dp[i + 1][j - 1];
				} else {
					dp[i][j] = Math.min(dp[i + 1][j], dp[i][j - 1]) + 1;
				}
			}
		}
		return dp;
	}


```



[[邮局选址问题]]
> 预处理数组record\[i\]\[j\]:   
 i...j 范围上建一个邮局, 总距离最小多少  
用来加速的流程, 只用上半部分, 表的下半区完全不用求, 不存在  
复杂度降到N^2
```java
	public static int[][] getRecord(int[] arr) {
		int N = arr.length;
		int[][] record = new int[N][N];
		for (int L = 0; L < N; L++) {
			// R < L 不需要填
			// record[L][R] L==R 0
			for (int R = L + 1; R < N; R++) {
				record[L][R] = record[L][R - 1] + arr[R] - arr[(L + R) >> 1];
			}
		}
		return record;
	}
```