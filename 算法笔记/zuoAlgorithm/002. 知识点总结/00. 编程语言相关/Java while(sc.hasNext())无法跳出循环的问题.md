# # Java while(sc.hasNext())无法跳出循环的问题

---
https://blog.csdn.net/weixin_44874269/article/details/114368278

当我们牛客网在线编程时 while(sc.hasNext()) 可以自己跳出循环，而在本地IDEA中使用while(sc.hasNext())是没有办法退出循环

```java

public static void main(String[] args) {  
    Scanner sc = new Scanner(System.in);  
    while (sc.hasNext()) {  
        int m = sc.nextInt();  
        int n = sc.nextInt();  
        System.out.printf("m: %s, n: %s.\n", m, n);  
    }  
    sc.close();  
}

// 使用输入quit退出
public static void main(String[] args) {
    Scanner sc = new Scanner(System.in);
    while (!sc.hasNext("quit")) { // 跳出循环
        int m = sc.nextInt();
        int n = sc.nextInt();
        System.out.printf("m: %s, n: %s.\n", m, n);
    }
    sc.close();
}

```