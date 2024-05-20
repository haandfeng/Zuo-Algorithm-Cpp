## Basecase 返回特定值及设置无效解的技巧

体系学习班19

---

### [[背包中有多少种零食放法]]  
表达无方案, 和方案数是0, 在此问题是统一的, 可以返回0, 做优化   
如果某些问题, 无方案-1, 和方案数是0是不同含义, 你最好分开的, 自己人为判断-1这件事情

```java
// 从左往右的经典模型
// 还剩的容量是rest，arr[index...]自由选择，
// 返回选择方案
// index ： 0～N
// rest : 0~w
public static int process(int[] arr, int index, int rest) {
    if (rest < 0) { // 没有容量了
        // -1 无方案的意思
        return -1;
    }
    // rest>=0,
    if (index == arr.length) { // 无零食可选
        return 1;
    }
    // rest >=0
    // 有零食index
    // index号零食，要 or 不要
    // index, rest
    // (index+1, rest)
    // (index+1, rest-arr[i])
    int next1 = process(arr, index + 1, rest); // 不要
    int next2 = process(arr, index + 1, rest - arr[index]); // 要
    return next1 + (next2 == -1 ? 0 : next2);
}
```

如果某个问题是, 你做出某种选择的情况下求最小值
你返回0
如果你怎么选是没有最小值的, 有可能进入一个无效的决策中, 这个最小值压根就不应该有, 
你就不应该拿这个最小值做决策, 这时候能不能返回0?  
不行, 因为你返0, 在自己含义是最小值你不用的意思, 你上游不知道这个, 会以为你返回0就是表达有效的

遇到具体问题的时候, 你明确有无效的东西要跟它返回0分开就分开写, 这道题可以统一
需要注意的是:   
但是关于-1的设置问题, 再解决其它问题的时候, 一定要拎的清除一点

### [[背包能装下最多的价值]]   


![[Pasted image 20210202145948.png]]


![[Pasted image 20210201114034.png]]

当你设置一个无效解, 在上游用它的时候就得提前判断, 有效你才能用, 无效你直接略过它

