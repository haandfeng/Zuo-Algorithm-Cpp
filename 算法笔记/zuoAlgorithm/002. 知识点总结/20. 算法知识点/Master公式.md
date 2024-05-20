# Master公式

主定理

---


形如:

$T(N) = a * T(N/b) + O(N^d)$(其中的a、b、d都是常数)  

Here n is the size of an input problem, a is the number of subproblems in the recursion, andb is the factor by which the subproblem size is reduced in each recursive call (b>1). 

的递归函数，可以直接通过Master公式来确定时间复杂度  
如果log(b,a) < d，复杂度为 $O(N^d)$  
如果 log(b,a) > d，复杂度为 O(N^log(b,a))  
如果 log(b,a) == d，复杂度为$O(N^d  * logN)$  


>master公式要求每个部分的复杂度一致才可以求解, 复杂度不一致无法使用, 比如:
$T(N) = T (1/3N) + T(2/3N) + O(N)$不可以使用公式求解



- [[根据要求构造出一个长度为M的数组]] , 复杂度 O(N)

ref: [master公式求时间复杂度](https://www.cnblogs.com/caijilin/p/15259392.html)