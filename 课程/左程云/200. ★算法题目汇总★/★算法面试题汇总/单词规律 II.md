# 单词规律 II

291.单词规律 II

[M]

---

https://leetcode-cn.com/problems/word-pattern-ii/

给你一种规律 pattern 和一个字符串 str，请你判断 str 是否遵循其相同的规律。

这里我们指的是 完全遵循，例如 pattern 里的每个字母和字符串 str 中每个 非空 单词之间，存在着 双射 的对应规律。双射 意味着映射双方一一对应，不会存在两个字符映射到同一个字符串，也不会存在一个字符分别映射到两个不同的字符串。
 
```
示例 1：

输入：pattern = "abab", s = "redblueredblue"
输出：true
解释：一种可能的映射如下：
'a' -> "red"
'b' -> "blue"
示例 2：

输入：pattern = "aaaa", s = "asdasdasdasd"
输出：true
解释：一种可能的映射如下：
'a' -> "asd"
示例 3：

输入：pattern = "abab", s = "asdasdasdasd"
输出：true
解释：一种可能的映射如下：
'a' -> "a"
'b' -> "sdasd"
注意 'a' 和 'b' 不能同时映射到 "asd"，因为这里的映射是一种双射。
示例 4：

输入：pattern = "aabb", s = "xyzabcxzyabc"
输出：false
 

提示：

0 <= pattern.length <= 20
0 <= s.length <= 50
pattern 和 s 由小写英文字母组成
```
