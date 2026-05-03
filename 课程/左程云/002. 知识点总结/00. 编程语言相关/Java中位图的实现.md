# Java语言中位图的实现

---

## 什么是[[位图]](bitmap)?
bitmap --> 用bit(位) 实现 map 的功能   


一个整数int, 4字节, 32bit, 实现一个bit数组, 每个位置占一个bit, 表示两种状态, 0, 1
可以用基础数组实现位图的结构

### 位图实际使用的空间
如果让位图数组从 0~m-1位置, m长度, 所需要的字节: m/8

### 数组实现
用基础类型数组是可以拼出位数组的

```java
public static void main(String[] args) {

    // 3200 bit bit[0~3199]
    int[] arr = new int[100];
    // arr[0] -> 32 bit 0(int)
    // arr[1] -> 32 bit 0(int)

    int position = 453;

    System.out.println(453/32);
    System.out.println(453%32);
    // arr[14] -> 010101000101 0 11010
    // 010101000101 0 11010 >> 5

    // 提取
    int status = ((arr[453/32] >> (453%32)) & 1);

    // 453 设置为1
    arr[453/32] = arr[453/32] | (1<<(453%32));

}

```

```java
public static class BitMap {

    private long[] bits;

    public BitMap(int max) {
        // (max + 64) >> 6 --> (max + 64) / 64
        // 一共要准备的整数个数
        bits = new long[(max + 64) >> 6];
    }

    public void add(int num) {
        // num >> 6 --> num / 64 --> 哪个正数
        // num % 64 --> num & 63
        bits[num >> 6] |= (1L << (num & 63));
    }

    public void delete(int num) {
        bits[num >> 6] &= ~(1L << (num & 63));
    }

    public boolean contains(int num) {
        return (bits[num >> 6] & (1L << (num & 63))) != 0;
    }
}

```

如果想表示 -10~10范围的数, 可以统一加10, 用 0~20的数表示 -10 ~ 10的数  

#### 为什么num % 64 可以用 num & 63 表示
因为 模 操作相当于 位运算比较慢

63 是 0000000000000000000000000000000000000000000000000000000000111111  
当num是一个具体的数字, 
num & 63 相当于把63的1前面全部搞成0 了, 剩下的东西就等同于 num % 64

只适用于num 模 2^n 方的情境下

#### 左移操作中一定使用1L, 而不能用1
`bits[num >> 6] &= ~(1L << (num & 63));`

比如: 1 << 42, 不加L, 会认为1是个整数, 只有32位, 就拿不到正确结果了



### 整形数组最多申请的长度
整形数组最多申请的长度 21亿
能表示的bit: 21亿\*32
```java
int[] arr1 = new int[Integer.MAX_VALUE];

long[] arr2 = new long[9999999999]; // 错误
```
如果还想表示更多的bit怎么办?
用long类型, 能表示的bit: 21亿\*64
如果需要比long类型还要多怎么办?
=>
二维矩阵描述位图


### 二维数组表示
如果下标不够用了, 总可以通过这样的表述表达下更多的bit

```java
// 320000bit
// matrix[0] -> 100 3200 bit
int[][] matrix = new int[100][100];

int pos = 170039;
// matrix[53]  170039 - (3200 * 53)   /32   %32
//             170039 % 3200          /32   %32
// 170039 / 3200

```



