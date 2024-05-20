# bfprt代码

#bfprt 

---

```java

// 利用bfprt算法求解，是O(N)的过程
// 在arr上，找到第K小的数，并返回。K范围是[1,N]，范围不是[0,N-1]
// 对你来讲，它可能永远不会被你想起，但确实本题最优解的算法原型
public static int getMinKthByBFPRT(int[] arr, int K) {
    return select(arr, 0, arr.length - 1, K - 1);
}

public static int select(int[] arr, int begin, int end, int i) {
    if (begin == end) {
        return arr[begin];
    }
    int pivot = medianOfMedians(arr, begin, end);
    int[] pivotRange = partition(arr, begin, end, pivot);
    if (i >= pivotRange[0] && i <= pivotRange[1]) {
        return arr[i];
    } else if (i < pivotRange[0]) {
        return select(arr, begin, pivotRange[0] - 1, i);
    } else {
        return select(arr, pivotRange[1] + 1, end, i);
    }
}

public static int medianOfMedians(int[] arr, int begin, int end) {
    int num = end - begin + 1;
    int offset = num % 5 == 0 ? 0 : 1;
    int[] mArr = new int[num / 5 + offset];
    for (int i = 0; i < mArr.length; i++) {
        int beginI = begin + i * 5;
        int endI = beginI + 4;
        mArr[i] = getMedian(arr, beginI, Math.min(end, endI));
    }
    return select(mArr, 0, mArr.length - 1, mArr.length / 2);
}

public static int[] partition(int[] arr, int begin, int end, int pivotValue) {
    int small = begin - 1;
    int cur = begin;
    int big = end + 1;
    while (cur != big) {
        if (arr[cur] < pivotValue) {
            swap(arr, ++small, cur++);
        } else if (arr[cur] > pivotValue) {
            swap(arr, cur, --big);
        } else {
            cur++;
        }
    }
    int[] range = new int[2];
    range[0] = small + 1;
    range[1] = big - 1;
    return range;
}

public static int getMedian(int[] arr, int begin, int end) {
    insertionSort(arr, begin, end);
    int sum = end + begin;
    int mid = (sum / 2) + (sum % 2);
    return arr[mid];
}

public static void insertionSort(int[] arr, int begin, int end) {
    for (int i = begin + 1; i != end + 1; i++) {
        for (int j = i; j != begin; j--) {
            if (arr[j - 1] > arr[j]) {
                swap(arr, j - 1, j);
            } else {
                break;
            }
        }
    }
}

public static void swap(int[] arr, int index1, int index2) {
    int tmp = arr[index1];
    arr[index1] = arr[index2];
    arr[index2] = tmp;
}


```