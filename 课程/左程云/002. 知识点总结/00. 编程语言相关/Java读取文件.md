# Java读取文件

---

https://www.cnblogs.com/hkgov/p/14707726.html


```java

void testReadFile1() throws IOException {

   String fileName = "D:\\data\\test.txt";

   try (Scanner sc = new Scanner(new FileReader(fileName))) {
      while (sc.hasNextLine()) {  //按行读取字符串
         String line = sc.nextLine();
         System.out.println(line);
      }
   }
}
```