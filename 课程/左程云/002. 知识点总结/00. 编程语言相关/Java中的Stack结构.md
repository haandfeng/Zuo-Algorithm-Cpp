# Java中的Stack结构


```java
// Stack继承自Vector，使用简单数组存储结构
Stack<Integer> stack = new Stack<>();
stack.push(1);
stack.push(1);
stack.push(1);
while(!stack.isEmpty()) {
    System.out.println(stack.pop());
}
```

Java里自己的实现是比较慢的, 栈实现的东西, 如果想优化常数实际, 别用这个结构, 
用LinkedList实现栈的功能: 每次都从尾部进, 从尾部出
就用双端队列替代了栈结构了, 还能快点

```java
// 可以用LinkedList头部追加头部弹出, 或者尾部追加尾部弹出两种方式实现
LinkedList<Integer> stack = new LinkedList<>();
stack.addFirst(1);
stack.addFirst(2);
stack.addFirst(3);

while(!stack.isEmpty()) {
    System.out.println(stack.pollFirst());
}
```

```java
// LinkedList里的push + pop 是数组头部追加头部弹出的方式
LinkedList<Integer> stack = new LinkedList<>();
stack.push(1);
stack.push(2);
stack.push(3);

while(!stack.isEmpty()) {
    System.out.println(stack.pop());
}
```


如果你明确知道你数组的长度, 比如假如是100, 拿数组结构替代是最快的
```java
int[] stack3 = new int[100];
int index = 0;
// 加入
stack3[index++] = 1;
stack3[index++] = 2;
stack3[index++] = 3;
// 弹出
System.out.println(stack3[--index]);
System.out.println(stack3[--index]);
System.out.println(stack3[--index]);
```



```java

System.out.println("===========Start==========");

testTime = 1000000;
Stack<Integer> stack4 = new Stack<>();
start = System.currentTimeMillis();
for(int i = 0; i < testTime; i++) {
    stack4.add(i);
}
while(!stack4.isEmpty()) {
    stack4.pop();
}
end = System.currentTimeMillis();
System.out.println(end-start);


int[] stack6 = new int[testTime];
start = System.currentTimeMillis();
int index = 0;
for(int i = 0; i < testTime; i++) {
    stack6[index++] = i;
}
while(index != 0) {
    int a = stack6[--index];
}
end = System.currentTimeMillis();
System.out.println(end-start);
System.out.println("===========End============");

```


```text
===========Start==========
36
5
===========End============
```

用数组实现一些Java中自有的数据结构来节省常数时间