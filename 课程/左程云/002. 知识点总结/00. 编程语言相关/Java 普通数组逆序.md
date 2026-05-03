# Java 普通数组逆序

---
普通数组只提供正序排序, 不能定义比较器, 无法逆序

```java
public static void main(String[] args) {  
    int[] arr = new int[]{3, 5, 7, 1, 3, 4, 2};  
    Arrays.sort(arr);  
    int i = 0, j = arr.length - 1;  
    while (i < j) {  
        swap(arr, i++, j--);  
    }  
    System.out.println(Arrays.toString(arr));  
}  
  
private static void swap(int[] arr, int i, int j) {  
    int tmp = arr[i];  
    arr[i] = arr[j];  
    arr[j] = tmp;  
}

```