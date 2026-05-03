# Java 打印数组

---

一维数组用 Arrays.toString() 方法，多维数组用 Arrays.deepToString() 方法

```java

public static void main(String[] args) {
    int[] arr = new int[]{0,1,2,3,4,5,6,7,8,9};
    //使用Array.toString()
    System.out.println(Arrays.toString(arr));
}

public static void main(String[] args) {
    int[][] arr = new int[][] {{1,2},{2,3},{3,4},{4,5},{5,6}};// 二维数组
    //使用Arrays.deepToString()
    System.out.println(Arrays.deepToString(arr));
}


```

https://blog.csdn.net/weixin_51631278/article/details/124044484