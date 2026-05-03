# LRU缓存机制

---

LRU是Least Recently Used的缩写，即最近最少使用
找时间操作距离现在最远的删掉, 加入新纪录
无论get, put都更新时间, 扔掉最远的没有操作的key
经典的只有put, get, 要求时间复杂度O(1)  
当然也可以加入delete



146. [[LRU内存替换算法的实现]]  [M]

https://leetcode-cn.com/problems/lru-cache