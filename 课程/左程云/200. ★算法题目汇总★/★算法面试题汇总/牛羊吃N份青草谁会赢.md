# 牛羊吃N份青草谁会赢
#打表法

---

给定一个正整数N，表示有N份青草统一堆放在仓库里有一只牛和一只羊，牛先吃，羊后吃，它俩轮流吃草不管是牛还是羊，每一轮能吃的草量必须是：  
1，4，16，64…(4的某次方)  
谁最先把草吃完，谁获胜假设牛和羊都绝顶聪明，都想赢，都会做出理性的决定根据唯一的参数N，返回谁会赢  

---

```java
// n份青草放在一堆
// 先手后手都绝顶聪明
// string "先手" "后手"
public static String winner1(int n) {
    // 0  1  2  3 4
    // 后 先 后 先 先
    if (n < 5) { // base case
        return (n == 0 || n == 2) ? "后手" : "先手";
    }
    // n >= 5 时
    int base = 1; // 当前先手决定吃的草数
    // 当前是先手在选
    while (base <= n) {
        // 当前一共n份草，先手吃掉的是base份，n - base 是留给后手的草
        // 母过程 先手 在子过程里是 后手
        if (winner1(n - base).equals("后手")) {
            return "先手";
        }
        if (base > n / 4) { // 防止base*4之后溢出
            break;
        }
        base *= 4;
    }
    return "后手";
}

```

==>

```java
public static String winner2(int n) {
    if (n % 5 == 0 || n % 5 == 2) {
        return "后手";
    } else {
        return "先手";
    }
}
```