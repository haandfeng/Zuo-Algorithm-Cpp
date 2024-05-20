# Java中的值传递跟引用传递


## 基本类型和引用类型在内存中的保存
https://www.cnblogs.com/binyue/p/3862276.html
https://blog.csdn.net/bjweimengshu/article/details/79799485

Java中数据类型分为两大类，基本类型和对象类型。 
相应的，变量也有两种类型：基本类型和引用类型。  
基本类型的变量保存原始值，即它代表的值就是数值本身；
而引用类型的变量保存引用值，"引用值"指向内存空间的地址，代表了某个对象的引用，而不是对象本身，
对象本身存放在这个引用值所表示的地址的位置。

基本类型包括：byte,short,int,long,char,float,double,Boolean,returnAddress，
引用类型包括：类类型，接口类型和数组。

相应的，变量也有两种类型：基本类型和引用类型。

（1）基本数据类型传值，对形参的修改不会影响实参；  
（2）引用类型传引用，形参和实参指向同一个内存地址（同一个对象），所以对参数的修改会影响到实际的对象；   
（3）String, Integer, Double等immutable的类型特殊处理，可以理解为传值，最后的操作不会修改实参对象。  

## Java方法参数中的值传递和引用传递
https://www.cnblogs.com/lingyejun/p/11028808.html

在Java方法中参数列表有两种类型的参数，基本类型和引用类型。

基本类型：值存放在局部变量表中，无论如何修改只会修改当前栈帧的值，方法执行结束对方法外不会做任何改变；此时需要改变外层的变量，必须返回主动赋值。

引用数据类型：指针存放在局部变量表中，调用方法的时候，副本引用压栈，赋值仅改变副本的引用。但是如果通过操作副本引用的值，修改了引用地址的对象，此时方法以外的引用此地址对象当然被修改。（两个引用，同一个地址，任何修改行为2个引用同时生效）。

这两种类型都是将外面的参数变量拷贝一份到局部变量中，基本类型为值拷贝，引用类型就是将引用地址拷贝一份。


结论：当方法参数为基本类型时，是将外部变量值拷贝到局部变量中而进行逻辑处理的，故方法是不能修改原基本变量的。

对于引用类型的方法参数，会将外部变量的引用地址，复制一份到方法的局部变量中，两个地址指向同一个对象。所以如果通过操作副本引用的值，修改了引用地址的对象，此时方法以外的引用此地址对象也会被修改。（两个引用，同一个地址，任何修改行为2个引用同时生效）。

ref:
https://www.cnblogs.com/xiximayou/p/12040285.html


### [[Java中Integer包装类的的128陷阱]]


## More...

[[收集达标路径和#什么时候需要考虑恢复现场]]

![[Pasted image 20210203205557.png]]

### 为什么pre = pre.next的时候, head没有动?

![[Pasted image 20210203223850.png]]

内存里建出一个Node, pre指向这块内存     
pre 指向一个内存区域, head也指向这块内存区域    

pre作为返回值(等号右边)的时候: 指的是它背后的东西    
当一个引用是等号左边, 是一个赋值的时候, 表示我指向谁    
后面pre自己动, Head还是指向这块内存区域, 不会动    
不是Head 指向pre, 而是指向pre背后的东西   


比如二叉树前序遍历
```java

public static void pre(Node head) {
    System.out.print("pre-order: ");
    if (head != null) {
        Stack<Node> stack = new Stack<Node>();
        stack.add(head);
        while (!stack.isEmpty()) {
            head = stack.pop();
            System.out.print(head.value + " ");
            if (head.right != null) {
                stack.push(head.right);
            }
            if (head.left != null) {
                stack.push(head.left);
            }
        }
    }
    System.out.println();
}

```

输入参数为链表头结点
