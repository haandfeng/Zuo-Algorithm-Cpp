# Markdown语法相关

---


### 分页

将这个插入到你要分页的地方
```
<div STYLE="page-break-after: always;"></div>  
```




### Markdown公式记法

$\Large x = \frac{-b \pm \sqrt{b^2-4ac} }{2a}$

$(log_{10}N)^2$

$tmp1/10 == 10^{k-2}$

$C_n^2$  

$\frac{expr1}{expr2}$

$\sqrt[x]{y}$  
$\sqrt{2}$

$\sum \frac{1}{i^2}$

$$\begin{matrix}
1&0&0\\
0&1&0\\
0&0&1\\
\end{matrix}$$



$$\begin{bmatrix}
{a_{11}}&{a_{12}}&{\cdots}&{a_{1n}}\\
{a_{21}}&{a_{22}}&{\cdots}&{a_{2n}}\\
{\vdots}&{\vdots}&{\ddots}&{\vdots}\\
{a_{m1}}&{a_{m2}}&{\cdots}&{a_{mn}}\\
\end{bmatrix}$$


$$\begin{cases}
a_1x+b_1y+c_1z=d_1\\
a_2x+b_2y+c_2z=d_2\\
a_3x+b_3y+c_3z=d_3\\
\end{cases}$$

$$sum=\begin{cases} L == 0 & H[R]\\
L != 0 & H[R] = H[L-1]\end{cases}$$


$$\begin{cases}a & x = 0\\ 
b & x > 0\end{cases}$$

### 大小
```text
$\tiny 萌萌哒$

$\scriptsize 萌萌哒$

$\small 萌萌哒$

$\normalsize 萌萌哒(正常)$

$\large 萌萌哒$

$\Large 萌萌哒$

$\huge 萌萌哒$

$\Huge 萌萌哒$
```


https://zhuanlan.zhihu.com/p/95886235


## 文本块引用
输入 \[\[^\]\] 本文的块,  \[\[^^\]\] 引用全库的块


## tag query
```
\```query

tag:字典序

\```
```


## tag nest
```
#动态规划/一个样本做行一个样本做列
```

## Example
 Comments will not show up in preview
 
 %%
 comments go here
 
 ## AAA
 
 %%


## 内嵌
https://publish.obsidian.md/help/How+to/Embed+files

![[XXXX.pdf#page=number]]


<iframe
    border=0
    frameborder=0
    height=250
    width=550  
    src="https://twitframe.com/show?url=https%3A%2F%2Ftwitter.com%2Fjack%2Fstatus%2F20">
</iframe>

<iframe src="https://player.bilibili.com/player.html?aid=586848024&bvid=BV1Kz4y1m74W&cid=300166684&page=1" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> 
</iframe>

$$
\begin{array}[b]{|c|c|} 
\hline 第一列 & 第二列& 第三列 \\ 
\hline 表格内容过长 & {当表格中要表达的内容比较多时，\\
页面上显示的表格会非常小，\\
看不清楚，如图所示，} & 所以要使用换行的办法 \\ 
\hline 
\end{array}
$$

