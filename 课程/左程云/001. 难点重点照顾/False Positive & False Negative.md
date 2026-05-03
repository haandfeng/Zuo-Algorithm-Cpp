# False Positive & False Negative

---

False Positive : False(检测模型不能成功地) Positive (判定出结果是Positive的)   
False Negative : False(检测模型不能成功地) Negative (判定出结果是Negative的)   
True Positive : True(检测模型成功地) Positive (判定出结果是Positive的)   
True Negative : True(检测模型成功地) Negative (判定出结果是Negative的)   

<br>

```text
比如在数据验证我们通过两数的相加和做一个checksum验证数据是否发生变化:   

False Negative 是指 check sum 认为是没问题的，但是可能存在问题。  
           这个是会发生的，比如 1,4 两个数据，check sum = 5，但是 2 + 3 也是 5。  
所以 check sum 相同不能证明数据一定没有发生变化。  
但是 check sum 不同就表示原始数据肯定发生了变化。  
因此 check sum 是存在 False Negative 但不存在 False Positive 的。
```
<br>

[[布隆过滤器]]说一个数存在有可能错误的，判断它不在肯定是对的   
把本来不存在[[布隆过滤器]]中的元素误判为存在的情况,我们把它叫做假阳性(False Positive)