# Manacher算法代码

#Manacher算法 

---


```java
public static int manacher(String s) {
    if (s == null || s.length() == 0) {
        return 0;
    }
    // "12132" -> "#1#2#1#3#2#"
    char[] str = manacherString(s);
    // 回文半径的大小
    int[] pArr = new int[str.length];
    int C = -1;
    // 讲述中：R代表最右的扩成功的位置。coding：最右的扩成功位置的，再下一个位置
    int R = -1;
    int max = Integer.MIN_VALUE;
    for (int i = 0; i < str.length; i++) {
        // R第一个违规的位置，i>= R
        // i位置扩出来的答案，i位置扩的区域，至少是多大。
        pArr[i] = R > i ? Math.min(pArr[2 * C - i], R - i) : 1;
        while (i + pArr[i] < str.length && i - pArr[i] > -1) {
            if (str[i + pArr[i]] == str[i - pArr[i]])
                pArr[i]++;
            else {
                break;
            }
        }
        if (i + pArr[i] > R) {
            R = i + pArr[i];
            C = i;
        }
        max = Math.max(max, pArr[i]);
    }
    return max - 1;
}

public static char[] manacherString(String str) {
    char[] charArr = str.toCharArray();
    char[] res = new char[str.length() * 2 + 1];
    int index = 0;
    for (int i = 0; i != res.length; i++) {
        res[i] = (i & 1) == 0 ? '#' : charArr[index++];
    }
    return res;
}

```
