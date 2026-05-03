# ICPC风格
---

不是LeetCode那种自己填函数的, 是需要自己处理输入输出的, 也叫ACM竞赛风格

```
public static void main(String[] args) {
	Scanner sc = new Scanner(System.in);
	int N = sc.nextInt();
	int M = sc.nextInt();
	int[][] matrix = new int[N][M];
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < M; j++) {
			matrix[i][j] = sc.nextInt();
		}
	}
	int ans = cherryPickup(matrix);
	System.out.println(ans);
	sc.close();
}

```

ref: [[最大路径和]]

```
public static void main(String[] args) {  
   Scanner sc = new Scanner(System.in);  
   int N = sc.nextInt();  
   int K = sc.nextInt();  
   int[] arr1 = new int[N];  
   int[] arr2 = new int[N];  
   for (int i = 0; i < N; i++) {  
      arr1[i] = sc.nextInt();  
   }  
   for (int i = 0; i < N; i++) {  
      arr2[i] = sc.nextInt();  
   }  
   int[] topK = topKSum(arr1, arr2, K);  
   for (int i = 0; i < K; i++) {  
      System.out.print(topK[i] + " ");  
   }  
   System.out.println();  
   sc.close();  
}
```

ref: [[两个有序数组相加和的Topk问题]]

[牛客网编程常见输入输出](https://blog.csdn.net/opencvzsefv/article/details/108177036) python + cpp