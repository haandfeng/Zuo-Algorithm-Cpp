# 判断字符串aim是否是str1和str2交错组成

#样本对应模型 
[M]

---

https://leetcode.com/problems/interleaving-string/


给定三个字符串str1、str2和aim，如果aim包含且仅包含来自str1和str2的所有字符， 而且在aim中属于str1的字符之间保持原来在str1中的顺序，属于str2的字符之间保持 原来在str2中的顺序，那么称aim是str1和str2的交错组成。实现一个函数，判断aim是 否是str1和str2交错组成
【举例】 str1="AB"，str2="12"。那么"AB12"、"A1B2"、"A12B"、"1A2B"和"1AB2"等都是 str1 和 str2 的 交错组成

---
>三个字符串对应一张二维表