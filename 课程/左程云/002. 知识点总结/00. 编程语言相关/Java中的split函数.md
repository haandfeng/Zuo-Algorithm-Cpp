# Java中的split函数

#硬记

---
ref: https://m.runoob.com/java/java-string-split.html

split() 方法根据匹配给定的正则表达式来拆分字符串。
注意： . 、 $、 | 和 * 等转义字符，必须得加 `\\`。
注意：多个分隔符，可以用 | 作为连字符。

split函数中  `\\` 表示转义, 如果按`\`切分, 需要写成`split("\\\\")`

```java
public class Split {

    public static void split1() {

        // \\s表示   空格,回车,换行等空白符,
        // +号表示一个或多个的意思\\s+表示已一个或多个空白为规则
        String s1 = "abc abc   ,   abca abc";
        String s2 = ", , , ,        a, eaefa";
        // 原因要分割出字符串中的空格，但是当空格个数多余一个时候
        // 就会默认分隔第一个，紧接后面的空格便会当成一个普通字符
        String [] str = s2.split(" ");
        for(String s : str)
        {
            System.out.println(s);
        }
        System.out.println("str数组的长度是:"+str.length);
    }

    public static void split2() {

        // \\s表示   空格,回车,换行等空白符,
        // +号表示一个或多个的意思\\s+表示已一个或多个空白为规则
        String s1 = "abc abc   ,   abca abc";
        String s2 = ", , , ,        a, eaefa";

        String [] str = s2.split("\\s+");
        for(String s : str)
        {
            System.out.println(s);
        }
        System.out.println("str数组的长度是:"+str.length);
    }

    public static void main(String[] args) {
        split1();
        System.out.println("========================");
        split2();
    }
}

```java
String[] words = article.split("\\s+"); // NOTE: 多个空格的切分技巧
```


```java
public static void main(String[] args) {  
  
    // "a\b\c"  
    String test = "a\\b\\c"; // 两个斜杠表示一个斜杠本身  
  
    // "a\b\c" 按斜杠切分, \ 是转义字符, 所以前面得加\\
    String[] arr = test.split("\\\\");  
    // String[] arr = test.split("\\\\", 2); // 分割为 a  b\c 两组  
    for (String str : arr) {  
        System.out.println(str);  
    }  
}

```


- [[打印目录结构]]