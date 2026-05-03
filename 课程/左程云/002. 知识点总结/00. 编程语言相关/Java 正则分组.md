# Java 正则分组

---

https://www.jianshu.com/p/a9b76d082eeb


```java
public static void eg3() {
    String input = "GOCB02组网口B链路中断\n";
    String regex = "^(GOCB)(\\d+)(组网口)(A|B)(链路中断)$";
    Pattern pattern = Pattern.compile(regex);
    Matcher matcher = pattern.matcher(input);
    if (matcher.find()) {
        System.out.println("X:"+matcher.group(2));
        System.out.println(input.replaceAll(regex, "$1" + ReplaceArg + "$3$4$5"));
    } else {
        System.out.println("not match");
    }
}
```
