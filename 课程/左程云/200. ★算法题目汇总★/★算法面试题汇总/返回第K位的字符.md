# 返回第K位的字符
#网易 

---
```
规定：

L[1]对应a，L[2]对应b，L[3]对应c，...，L[25]对应y  
S1 = a  
S(i) = S(i-1) + L[i] + reverse(invert(S(i-1)));  

解释invert操作：  
S1 = a  
S2 = aby  
假设invert(S(2)) = 甲乙丙  
a + 甲 = 26, 那么 甲 = 26 - 1 = 25 -> y  
b + 乙 = 26, 那么 乙 = 26 - 2 = 24 -> x  
y + 丙 = 26, 那么 丙 = 26 - 25 = 1 -> a  
如上就是每一位的计算方式，所以invert(S2) = yxa  
所以S3 = S2 + L[3] + reverse(invert(S2)) = aby + c + axy = abycaxy  
invert(abycaxy) = yxawyba, 再reverse = abywaxy  
所以S4 = abycaxy + d + abywaxy = abycaxydabywaxy  
直到S25结束  
给定两个参数n和k，返回Sn的第k位是什么字符，n从1开始，k从1开始  
比如n=4，k=2，表示S4的第2个字符是什么，返回b字符  
```

ref: 1545.[[找出第 N 个二进制字符串中的第 K 位]] [M]