#  java 中Integer包装类的的128陷阱

---

## java 中Integer包装类的的128陷阱
https://blog.csdn.net/qq_41644715/article/details/99981403

java开发者认为
每次都要开辟新空间会占用大量的资源，因此他们规定在-128~127（因为Java设计者认为大家对数的使用大多在100以内）之间的Integer类型的变量，直接指向常量池中的缓存地址，不会new开辟出新的空间。 常量池存在于方法区，类加载时就有了。

在整数的包装类当中，在第一次创建 Integer 类的对象的时候，都会首先创建好缓存数组。当需要包装的值是在 IntegerCache 数组当中的元素的时候，就会返回数组当中的 Integer 对象。JVM 默认就会设置数组的范围为 -128 ~ 127 。除非设置 JVM 当中的 AutoBoxCacheMax 属性大小即可

```
private static class IntegerCache {
    // 这是 IntegerCache 初始化的最小值，注意跟虚拟机当中的属性没有任何关系。
    static final int low = -128;
    // 这是 IntegetCache 初始化的最大值，注意他是跟虚拟机的配置有直接的关系。
    static final int high;
    // 这是用来保存整数缓存池当中的元素
    static final Integer[] cache;
    static {
        // high value may be configured by property
        int h = 127;
        // 获得 JVM 当中设置缓存当中最大的数值。如果没有设置，JVM 默认大小即为 127。设置方式：在 JVM 的 AutoBoxCacheMax=需要设置的缓冲区保存的最大值。
        String integerCacheHighPropValue = VM.getSavedProperty("java.lang.Integer.IntegerCache.high");
        if (integerCacheHighPropValue != null) {
            try {
                int i = parseInt(integerCacheHighPropValue);
                // 将从虚拟机当中获得的值与默认的最大值进行比较，筛选出里面的最大值
                i = Math.max(i, 127);
                // 取 i 与 Integer.MAX_VALUE - (-low) - 1 之间的最小值。因为这是保证缓存区的数组大小不大于 Integer.Max_Value。
                h = Math.min(i, Integer.MAX_VALUE - (-low) - 1);
            } catch (NumberFormatException nfe) {
                // 在 parseInt 方法当中如果转换失败，会捕捉到 NumberFormatException。对这个异常默认没有任何处理。
            }
        }
        high = h;
        // 创建缓存池当中的空间，取决于最大值与最小值之间的空间。*
        cache = new Integer[(high - low) + 1];
        int j = low;
        for(int k = 0; k < cache.length; k++)
            cache[k] = new Integer(j++);

        // 最后通过断言，检查缓冲区的上界至少是 127。
        assert IntegerCache.high >= 127;
    }

    private IntegerCache() {}
}
```