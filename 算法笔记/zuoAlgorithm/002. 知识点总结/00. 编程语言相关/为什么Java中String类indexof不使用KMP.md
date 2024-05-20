# 为什么Java中String类indexof不使用KMP


---

JDK的String类中的indexof方法的实现，仅仅只是用了暴力破解法，也就是最原始的实现，时间复杂度也到了O(n * m)。

JDK的编写者们认为大多数情况下，字符串都不长，使用原始实现可能代价更低。因为KMP和Boyer-Moore算法都需要预先计算处理来获得辅助数组，需要一定的时间和空间，这可能在短字符串查找中相比较原始实现耗费更大的代价。而且一般大字符串查找时，程序员们也会使用其它特定的数据结构，查找起来更简单。这有点类似于排除特定情况下的快速排序了。不同环境选择不同算法。


```java
/**
 * Code shared by String and StringBuffer to do searches. The
 * source is the character array being searched, and the target
 * is the string being searched for.
 *
 * @param   source       the characters being searched.
 * @param   sourceOffset offset of the source string.
 * @param   sourceCount  count of the source string.
 * @param   target       the characters being searched for.
 * @param   targetOffset offset of the target string.
 * @param   targetCount  count of the target string.
 * @param   fromIndex    the index to begin searching from.
 */
static int indexOf(char[] source, int sourceOffset, int sourceCount,
        char[] target, int targetOffset, int targetCount,
        int fromIndex) {
    if (fromIndex >= sourceCount) {
        return (targetCount == 0 ? sourceCount : -1);
    }
    if (fromIndex < 0) {
        fromIndex = 0;
    }
    if (targetCount == 0) {
        return fromIndex;
    }

    char first = target[targetOffset];
    int max = sourceOffset + (sourceCount - targetCount);

    for (int i = sourceOffset + fromIndex; i <= max; i++) {
        /* Look for first character. */
        if (source[i] != first) {
            while (++i <= max && source[i] != first);
        }

        /* Found first character, now look at the rest of v2 */
        if (i <= max) {
            int j = i + 1;
            int end = j + targetCount - 1;
            for (int k = targetOffset + 1; j < end && source[j]
                    == target[k]; j++, k++);

            if (j == end) {
                /* Found whole string. */
                return i - sourceOffset;
            }
        }
    }
    return -1;
}


```

http://stackoverflow.com/questions/19543547/why-jdks-string-indexof-does-not-use-kmp/

