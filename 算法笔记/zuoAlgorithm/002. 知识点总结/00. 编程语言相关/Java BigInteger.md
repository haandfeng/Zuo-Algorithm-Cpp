# Java BigInteger, BigDecimal

---

BigInteger和BigDecimal都是不可变(immutable)的，在进行每一步运算时，都会产生一个新的对象，由于创建对象会引起开销，
* 它们不适合于大量的数学计算，应尽量用long,float,double等基本类型做科学计算或者工程计算。
* 设计BigInteger和BigDecimal的目的是用来精确地表示大整数和小数，使用于在商业计算中使用。


BigDecimal有4个够造方法，其中的两个用BigInteger构造，另一个是用double构造，还有一个使用String构造。
* 应该避免使用double构造BigDecimal，因为：有些数字用double根本无法精确表示，传给BigDecimal构造方法时就已经不精确了。
* 比如，new BigDecimal(0.1)得到的值是0.1000000000000000055511151231257827021181583404541015625。
* 使用new BigDecimal("0.1")得到的值是0.1。因此，如果需要精确计算，用String构造BigDecimal，避免用double构造，尽管它看起来更简单！


equals()方法认为0.1和0.1是相等的，返回true，而认为0.10和0.1是不等的，结果返回false。
方法compareTo()则认为0.1与0.1相等，0.10与0.1也相等。所以在从数值上比较两个BigDecimal值时，应该使用compareTo()而不是 equals()。

另外还有一些情形，任意精度的小数运算仍不能表示精确结果。例如，1除以9会产生无限循环的小数 .111111...。
* 出于这个原因，在进行除法运算时，BigDecimal可以让您显式地控制舍入。

https://www.cnblogs.com/z-y-z/archive/2013/02/19/2917675.html