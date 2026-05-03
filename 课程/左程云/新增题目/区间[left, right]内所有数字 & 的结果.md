注意一个点：只要某一个数的1位有0，那么那一位一定为0；
如果 left!=right，那么right的最右边的1一定留不住(因为一旦减去1之后，最右边的1那一位就会变0)，所以删掉最左边的那一位1.
```java
// 给你两个整数 left 和 right ，表示区间 [left, right]// 返回此区间内所有数字 & 的结果  
// 包含 left 、right 端点  
// 测试链接 : https://leetcode.cn/problems/bitwise-and-of-numbers-range/
public class Code04_LeftToRightAnd {  
  
   public static int rangeBitwiseAnd(int left, int right) {  
      while (left < right) {  
         right -= right & -right;  
      }  
      return right;  
   }  
  
}
```