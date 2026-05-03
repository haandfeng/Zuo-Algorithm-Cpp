# 有序数组中找到>=num最左的位置

#二分法 

---

```java
// arr有序的，>=num 最左
public static int mostLeftNoLessNumIndex(int[] arr, int num) {
    if (arr == null || arr.length == 0) {
        return -1;
    }
    int L = 0;
    int R = arr.length - 1;
    int ans = -1;
    while (L <= R) {
        int mid = (L + R) / 2;
        if (arr[mid] >= num) {
            ans = mid;
            R = mid - 1;
        } else {
            L = mid + 1;
        }
    }
    return ans;
}

```

哪怕没有找到, ans没有更新 这个代码也是对的, 比如[1,2,3,4]数组中找100, 一直走else分支, ans没有被更新


