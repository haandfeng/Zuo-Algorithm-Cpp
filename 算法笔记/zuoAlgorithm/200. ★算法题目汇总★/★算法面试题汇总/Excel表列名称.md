# Excel表列名称

168.Excel表列名称

[E]

---

https://leetcode-cn.com/problems/excel-sheet-column-title/


```java
// 正常26进制是有0的, 比如(低位0~25, 高位1低位0即10代表一个26)  
// 本题每位1~26, 待转换数字num从1开始~N  
// 当num属于[1~26] % 26 = 1...25, 0(当前位值为26时为0)  
// 即高位的1代表低位的26, 而低位为26时, 相当于低位有一个高位的1  
// 正常26进制, 26在高一位, 伪26进制, 当mod值为0时, 代表当前为有26  
// 计算完后num - 26, 相当于从高一位减1  
// _(个26) _(1~26), 低位有一个高位的26的1  
public static String convertToTitle2(int columnNumber) {  
    StringBuilder sb = new StringBuilder();  
    while (columnNumber > 0) {  
        int rem = columnNumber%26;  
        if (rem == 0) {//如果余数是0，代表当前位为26, 修正余数为26  
 rem = 26;  
            columnNumber -= 26; // 低位有一个高位的26的1, 减去  
 // 相当于/26后得到的高位要 -1 }  
        sb.insert(0, (char)(rem - 1  + 'A'));  
        columnNumber /= 26;  
    }  
    return sb.toString();  
}

```