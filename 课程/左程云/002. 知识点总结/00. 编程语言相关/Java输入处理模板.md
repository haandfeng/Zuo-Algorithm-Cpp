# Java输入处理模板

---

ref: https://www.jianshu.com/p/11158dbc7bde

[[ICPC风格]]

## 简单输入

```java
import java.util.Scanner 
import java.io.BufferedInputStream;

public class SimpleInput {  
  
    public static void main(String[] args) {  
        long t1, t2;  
        Scanner sc1 = new Scanner(System.in);  
        // sc2用到缓冲流，读入更快  
        Scanner sc2 = new Scanner(new BufferedInputStream(System.in));  
  
        t1 = System.nanoTime();  
        String input1 = sc1.next();  
        t2 = System.nanoTime();  
  
        System.out.println("sc1:" + (t2 - t1));  
        System.out.println(input1);  
  
        //////////  
  
        t1 = System.nanoTime();  
        String input2 = sc2.next();  
        t2 = System.nanoTime();  
  
        System.out.println("sc2:" + (t2 - t1));  
        System.out.println(input2);  
    }  
  
}
```

```java
// 1,2,3,4,5
Scanner sc = new Scanner(System.in);  
String str = sc.next();  
  
String[] split = str.split(",");

```

## 字符串分割

```java
// 二者都能把一个或多个空格换成逗号
String ss = str.replaceAll(" +", ",");
String ss = str.replaceAll("\\s+", ",");


String[] sa = str.split(" +");
String[] sa = str.split("\\s+",);
```


