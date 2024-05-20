# Java 读写文件

---
#### 读文件

```java
// 从标准输入读取
Scanner input = new Scanner(System.in);

// 从文件中读取数据
File file = new File("文件路径");

//要指定文件的编码格式，否则可能读不了数据
Scanner sc = new Scanner(file，“GBK”);

while(sc.hasNext()){
    String line = sc.next();
    System.out.println(line);
}

```

Scanner中的next()方法读取一个由分隔符(默认空格)分割的字符串

nextLine()读取一个以行分隔符结束的行。

注意：行分隔符是由系统定义的，在Windows平台是\r\n,而在UNIX平台上是\n。

为了得到特定平台上的行分隔符，使用

`String line = System.getProperty("file.separator");`

#### 写文件

```java
String outFile = "/tmp/data.out";  
PrintWriter out = new PrintWriter(outFile);  
out.println(res);  
out.close();
```