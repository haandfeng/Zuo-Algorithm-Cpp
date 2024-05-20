# 最常使用的K个单词II
[H]

---
https://www.lintcode.com/problem/top-k-frequent-words-ii/

描述
在实时数据流中找到最常使用的k个单词.
实现TopK类中的三个方法:
TopK(k), 构造方法
add(word), 增加一个新单词
topk(), 得到当前最常使用的k个单词.

如果两个单词有相同的使用频率, 按字典序排名.
```
样例
```text
样例 1:
输入：
TopK(2)
add("lint")
add("code")
add("code")
topk()
输出：["code", "lint"]
解释：
"code" 出现两次并且 "lint" 出现一次， 它们是出现最频繁的两个单词。
样例 2:
输入：
TopK(1)
add("aa")
add("ab")
topk()
输出：["aa"]
解释：
"aa" 和 "ab" 出现 , 但是aa的字典序小于ab。
```