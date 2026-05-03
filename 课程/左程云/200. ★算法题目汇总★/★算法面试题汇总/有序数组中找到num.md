# 有序数组中找到num

#二分法 

---


```java
// arr保证有序
public static boolean find(int[] arr, int num) {
    if (arr == null || arr.length == 0) {
        return false;
    }
    int L = 0;
    int R = arr.length - 1;
    while (L <= R) {
        int mid = (L + R) / 2;
        if (arr[mid] == num) {
            return true;
        } else if (arr[mid] < num) {
            L = mid + 1;
        } else {
            R = mid - 1;
        }
    }
    return false;
}

```