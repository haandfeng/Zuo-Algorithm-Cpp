# Java 静态块构造块

---

静态块是在类第一次实例化的时候执行一次，且仅仅是执行一次。

构造块是每次类被实例化的时候都会被调用一次，它的作用是将所有对象进行统一初始化。表现形式：{ //代码}

执行顺序为 静态块 > 构造块 > 构造方法块 


```java
class A {  
    static {  
        System.out.println("A 静态块>>>>>");  
    }  
  
    {  
        System.out.println("A 构造块>>>>>");  
    }  
  
    A() {  
        System.out.println("A 无参构造方法>>>>>");  
    }  
  
    A(int a) {  
        System.out.println("A 有参构造方法>>>>>: " + a);  
    }  
}  
  
class B extends A {  
    static {  
        System.out.println("B 静态块>>>>>");  
    }  
  
    {  
        System.out.println("B 构造块>>>>>");  
    }  
  
    B() {  
        System.out.println("B 无参构造方法>>>>>");  
    }  
  
    B(int a) {  
        System.out.println("B 有参构造方法>>>>>: " + a);  
    }  
}

public static void main(String[] args) {  
     B test = new B();  
    System.out.println("+++++++++++++");  
     B test2 = new B(1);  
}
```

 程序执行了一次 new TestB() ，由于 TestB 类继承了 TestA 类，程序会先去执行 TestA 类的静态代码块， 然后执行 TestB 类的静态代码块，然后执行 TestA 类的构造代码块和构造方法，最后执行 TestB 类的构造代码块和构造方法。  