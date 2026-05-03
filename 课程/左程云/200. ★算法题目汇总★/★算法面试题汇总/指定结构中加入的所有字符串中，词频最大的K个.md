# 指定结构中加入的所有字符串中，词频最大的K个



---
请实现如下结构：
TopRecord{
public TopRecord(int K)  :  构造时事先指定好K的大小，构造后就固定不变了
public  void add(String str)  :   向该结构中加入一个字符串，可以重复加入
public  List<String> top() : 返回之前加入的所有字符串中，词频最大的K个
}
要求： 
add方法，复杂度O(log K);
top方法，复杂度O(K)
