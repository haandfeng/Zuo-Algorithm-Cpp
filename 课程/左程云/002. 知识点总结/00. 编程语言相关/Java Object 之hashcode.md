# Java Object 之hashcode

---
https://blog.csdn.net/qq_55272229/article/details/122890797
https://blog.csdn.net/JasonChen3318/article/details/100150009

Note: 自定义类对象, 数组对象(`int[]`) 取hashCode()就是地址, 基本类型hashCode() 不是地址, 需要包一层

1.hashcode 主要是为提高hash容器的效率，这是一种**根据对象内存**地址来决定一段整数编码
2.俩个引用如果指向同一对象，那么hash肯定会值一样
3.俩个引用如果指向不同对象，择hash值不一样
4.hash值是根据地址来计算，但是不能等价内存就是hash值
事实上有object类定义的hashcode 方法确实会针对不同对象返回不同整数，这是一般将内部地址转换（c++底层才能获取，因为java本身是在jvm上运行的）为整数
5.hashcode往往在集合中重写。


在Object中hashCode()返回的是对象的地址值  
tostring()方法会默认返回全类名(包名+类名) + @ + hashCode() 值的16 进制
https://www.cnblogs.com/atao-BigData/p/16054102.html


`getClass().getName+‘@’+Integer.toHexSting(hashCode)`

![[Pasted image 20220906213739.png|500]]


Arrays.hashCode 和  Arrays.toString  可以让你能用一个 数组类型做 key
ref: 

1439.[[有序矩阵中的第 k 个最小数组和]] [H]
1815.[[得到新鲜甜甜圈的最多组数]]  [H]