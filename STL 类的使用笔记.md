
# 1. set/ multiset/unorder 容器
## 1.1. 基本概念
所有元素都会在插入时自动被排序，底层通过二叉树实现

**set和multiset区别**：
- multiset允许容器中有重复的元素
- set不允许容器中有重复的元素
## 1.2. 构造,插入,删除,遍历
构造：
- `set<T> st;` //默认构造函数：
- `set(const set &st);` //拷贝构造函数

赋值：
- `set& operator=(const set &st);` //重载等号操作符

插入和删除：
- `insert(elem);` //在容器中插入元素。
- `clear();` //清除所有元素
- `erase(pos);` //删除pos迭代器所指的元素，返回下一个元素的迭代器。
- `erase(beg, end);` //删除区间\[beg,end)的所有元素 ，返回下一个元素的迭代器。 
- `erase(elem);` //删除容器中值为elem的元素。

代码示例:
通过 insert插入元素，可以用迭代器遍历
```c++
#include <set>

void printSet(set<int> & s)
{
	for (set<int>::iterator it = s.begin(); it != s.end(); it++)
	{
		cout << *it << " ";
	}
	cout << endl;
}

//插入和删除
void test01()
{
	set<int> s1;
	//插入
	s1.insert(10);
	s1.insert(30);
	s1.insert(20);
	s1.insert(40);
	printSet(s1);

	//删除
	s1.erase(s1.begin());
	printSet(s1);

	s1.erase(30);
	printSet(s1);

	//清空
	//s1.erase(s1.begin(), s1.end());
	s1.clear();
	printSet(s1);
}

int main() {
	test01();
}
```

## 1.3. set大小和交换
**函数原型：**
- `size();` //返回容器中元素的数目
- `empty();` //判断容器是否为空
- `swap(st);` //交换两个集合容器

```c++
#include <set>

void printSet(set<int> & s)
{
	for (set<int>::iterator it = s.begin(); it != s.end(); it++)
	{
		cout << *it << " ";
	}
	cout << endl;
}

//大小
void test01()
{

	set<int> s1;
	
	s1.insert(10);
	s1.insert(30);
	s1.insert(20);
	s1.insert(40);

	if (s1.empty())
	{
		cout << "s1为空" << endl;
	}
	else
	{
		cout << "s1不为空" << endl;
		cout << "s1的大小为： " << s1.size() << endl;
	}

}

//交换
void test02()
{
	set<int> s1;

	s1.insert(10);
	s1.insert(30);
	s1.insert(20);
	s1.insert(40);

	set<int> s2;

	s2.insert(100);
	s2.insert(300);
	s2.insert(200);
	s2.insert(400);

	cout << "交换前" << endl;
	printSet(s1);
	printSet(s2);
	cout << endl;

	cout << "交换后" << endl;
	s1.swap(s2);
	printSet(s1);
	printSet(s2);
}

int main() {

	test02();
}
```

## 1.4. set的查找和统计

**函数原型：**

- `find(key);` //查找key是否存在,若存在，返回该键的元素的迭代器；若不存在，返回set.end();
- `count(key);` //统计key的元素个数

```c++
#include <set>

//查找和统计
void test01()
{
	set<int> s1;
	//插入
	s1.insert(10);
	s1.insert(30);
	s1.insert(20);
	s1.insert(40);
	
	//查找
	set<int>::iterator pos = s1.find(30);

	if (pos != s1.end())
	{
		cout << "找到了元素 ： " << *pos << endl;
	}
	else
	{
		cout << "未找到元素" << endl;
	}

	//统计
	int num = s1.count(30);
	cout << "num = " << num << endl;
}

int main() {
	test01();
}
```

## 1.5. Set 插入重复数据
- set不可以插入重复数据，而multiset可以
- set插入数据的同时会返回插入结果，表示插入是否成功
- multiset不会检测数据，因此可以插入重复数据

```c++
#include <set>

//set和multiset区别
void test01()
{
	set<int> s;
	pair<set<int>::iterator, bool>  ret = s.insert(10);
	if (ret.second) {
		cout << "第一次插入成功!" << endl;
	}
	else {
		cout << "第一次插入失败!" << endl;
	}

	ret = s.insert(10);
	if (ret.second) {
		cout << "第二次插入成功!" << endl;
	}
	else {
		cout << "第二次插入失败!" << endl;
	}
    
	//multiset
	multiset<int> ms;
	ms.insert(10);
	ms.insert(10);

	for (multiset<int>::iterator it = ms.begin(); it != ms.end(); it++) {
		cout << *it << " ";
	}
	cout << endl;
}

int main() {
	test01();
}
```

## 1.6. set 排序
set默认会从很小到大排序，要改变需要自定义规则，如果是自定义类就没事
```c++
class MyCompare 
{
public:
	bool operator()(int v1, int v2) {
		return v1 > v2;
	}
};
	//指定排序规则，从大到小
	set<int,MyCompare> s2;
```

## 1.7. unorder_set和set
• std::set(红黑树)适用于：
1. 需要保持元素的顺序。
2. 需要进行范围查询（如 lower_bound 和 upper_bound）。
3. 不特别关心操作的常数时间性能。
4. 查找/插入/删除复杂度 O(log n) 
• std::unordered_set(哈希表) 适用于：
1. 只需要快速查找元素，而不关心顺序。
2. 对性能要求较高，尤其是查找、插入和删除操作。
3. 需要存储大量数据，并且哈希冲突不严重。
4. 平均 O(1)，最坏 O(n)

# 2. Pair对组
- 成对出现的数据，利用对组可以返回两个数据
**两种创建方式：**
- `pair<type, type> p ( value1, value2 );`
- `pair<type, type> p = make_pair( value1, value2 );`

```c++
#include <string>
//对组创建
void test01()
{
	pair<string, int> p(string("Tom"), 20);
	cout << "姓名： " <<  p.first << " 年龄： " << p.second << endl;

	pair<string, int> p2 = make_pair("Jerry", 10);
	cout << "姓名： " << p2.first << " 年龄： " << p2.second << endl;
}

int main() {
	test01();
}
```



# map/ multimap容器

## 基本概念

**简介：**
- map中所有元素都是pair
- pair中第一个元素为key（键值），起到索引作用，第二个元素为value（实值）
- 所有元素都会根据元素的键值自动排序
- 可以根据key值快速找到value值

map和multimap**区别**：
- map不允许容器中有重复key值元素
- multimap允许容器中有重复key值元素


## map 赋值构造 插入和删除

**构造：**
- `map<T1, T2> mp;` //map默认构造函数:
- `map(const map &mp);` //拷贝构造函数
**赋值：**
- `map& operator=(const map &mp);` //重载等号操作符
- 
**插入和删除:**
- `insert(elem);` //在容器中插入元素。
- `clear();` //清除所有元素
- `erase(pos);` //删除pos迭代器所指的元素，返回下一个元素的迭代器。
- `erase(beg, end);` //删除区间\[beg,end)的所有元素 ，返回下一个元素的迭代器。
- `erase(key);` //删除容器中值为key的元素。

```c++
#include <map>

void printMap(map<int,int>&m)
{
	for (map<int, int>::iterator it = m.begin(); it != m.end(); it++)
	{
		cout << "key = " << it->first << " value = " << it->second << endl;
	}
	cout << endl;
}

void test01()
{
	//插入
	map<int, int> m;
	//第一种插入方式
	m.insert(pair<int, int>(1, 10));
	//第二种插入方式
	m.insert(make_pair(2, 20));
	//第三种插入方式
	m.insert(map<int, int>::value_type(3, 30));
	//第四种插入方式
	m[4] = 40; 
	printMap(m);

	//删除
	m.erase(m.begin());
	printMap(m);
	// 按照key删除
	m.erase(3);
	printMap(m);

	//清空
	m.erase(m.begin(),m.end());
	m.clear();
	printMap(m);
}

int main() {

	test01();
}
```

## map大小和交换
- `size();` //返回容器中元素的数目
- `empty();` //判断容器是否为空
- `swap(st);` //交换两个集合容器

```c++
void printMap(map<int,int>&m)
{
	for (map<int, int>::iterator it = m.begin(); it != m.end(); it++)
	{
		cout << "key = " << it->first << " value = " << it->second << endl;
	}
	cout << endl;
}

void test01()
{
	map<int, int>m;
	m.insert(pair<int, int>(1, 10));
	m.insert(pair<int, int>(2, 20));
	m.insert(pair<int, int>(3, 30));

	if (m.empty())
	{
		cout << "m为空" << endl;
	}
	else
	{
		cout << "m不为空" << endl;
		cout << "m的大小为： " << m.size() << endl;
	}
}


//交换
void test02()
{
	map<int, int>m;
	m.insert(pair<int, int>(1, 10));
	m.insert(pair<int, int>(2, 20));
	m.insert(pair<int, int>(3, 30));

	map<int, int>m2;
	m2.insert(pair<int, int>(4, 100));
	m2.insert(pair<int, int>(5, 200));
	m2.insert(pair<int, int>(6, 300));

	cout << "交换前" << endl;
	printMap(m);
	printMap(m2);

	cout << "交换后" << endl;
	m.swap(m2);
	printMap(m);
	printMap(m2);
}

int main() {
	test01();
	test02();
```

## map查找和统计

- `find(key);` //查找key是否存在,若存在，返回该键的元素的迭代器；若不存在，返回set.end();
- `count(key);` //统计key的元素个数

```c++
#include <map>

//查找和统计
void test01()
{
	map<int, int>m; 
	m.insert(pair<int, int>(1, 10));
	m.insert(pair<int, int>(2, 20));
	m.insert(pair<int, int>(3, 30));

	//查找
	map<int, int>::iterator pos = m.find(3);

	if (pos != m.end())
	{
		cout << "找到了元素 key = " << (*pos).first << " value = " << (*pos).second << endl;
	}
	else
	{
		cout << "未找到元素" << endl;
	}

	//统计
	int num = m.count(3);
	cout << "num = " << num << endl;
}

int main() {
	test01();
}
```


## map容器排序

- map容器默认排序规则为 按照key值进行 从小到大排序，掌握如何改变排序规则
- 利用仿函数，可以改变排序规则

```c++
#include <map>

class MyCompare {
public:
	bool operator()(int v1, int v2) {
		return v1 > v2;
	}
};

void test01() 
{
	//默认从小到大排序
	//利用仿函数实现从大到小排序
	map<int, int, MyCompare> m;

	m.insert(make_pair(1, 10));
	m.insert(make_pair(2, 20));
	m.insert(make_pair(3, 30));
	m.insert(make_pair(4, 40));
	m.insert(make_pair(5, 50));

	for (map<int, int, MyCompare>::iterator it = m.begin(); it != m.end(); it++) {
		cout << "key:" << it->first << " value:" << it->second << endl;
	}
}
int main() {

	test01();

	system("pause");

	return 0;
}
```

#  常见的三种哈希结构

当我们想使用哈希法来解决问题的时候，我们一般会选择如下三种数据结构。

- 数组
- set （集合）
- map(映射)

这里数组就没啥可说的了，我们来看一下set。

在C++中，set 和 map 分别提供以下三种数据结构，其底层实现以及优劣如下表所示：

|集合|底层实现|是否有序|数值是否可以重复|能否更改数值|查询效率|增删效率|
|---|---|---|---|---|---|---|
|std::set|红黑树|有序|否|否|O(log n)|O(log n)|
|std::multiset|红黑树|有序|是|否|O(logn)|O(logn)|
|std::unordered_set|哈希表|无序|否|否|O(1)|O(1)|

std::unordered_set底层实现为哈希表，std::set 和std::multiset 的底层实现是红黑树，红黑树是一种平衡二叉搜索树，所以key值是有序的，但key不可以修改，改动key值会导致整棵树的错乱，所以只能删除和增加。

|映射|底层实现|是否有序|数值是否可以重复|能否更改数值|查询效率|增删效率|
|---|---|---|---|---|---|---|
|std::map|红黑树|key有序|key不可重复|key不可修改|O(logn)|O(logn)|
|std::multimap|红黑树|key有序|key可重复|key不可修改|O(log n)|O(log n)|
|std::unordered_map|哈希表|key无序|key不可重复|key不可修改|O(1)|O(1)|

std::unordered_map 底层实现为哈希表，std::map 和std::multimap 的底层实现是红黑树。同理，std::map 和std::multimap 的key也是有序的（这个问题也经常作为面试题，考察对语言容器底层的理解）。

当我们要使用集合来解决哈希问题的时候，优先使用unordered_set，因为它的查询和增删效率是最优的，如果需要集合是有序的，那么就用set，如果要求不仅有序还要有重复数据的话，那么就用multiset。

那么再来看一下map ，在map 是一个key value 的数据结构，map中，对key是有限制，对value没有限制的，因为key的存储方式使用红黑树实现的。

其他语言例如：java里的HashMap ，TreeMap 都是一样的原理。可以灵活贯通。

虽然std::set和std::multiset 的底层实现基于红黑树而非哈希表，它们通过红黑树来索引和存储数据。不过给我们的使用方式，还是哈希法的使用方式，即依靠键（key）来访问值（value）。所以使用这些数据结构来解决映射问题的方法，我们依然称之为哈希法。std::map也是一样的道理。

这里在说一下，一些C++的经典书籍上 例如STL源码剖析，说到了hash_set hash_map，这个与unordered_set，unordered_map又有什么关系呢？

实际上功能都是一样一样的， 但是unordered_set在C++11的时候被引入标准库了，而hash_set并没有，所以建议还是使用unordered_set比较好，这就好比一个是官方认证的，hash_set，hash_map 是C++11标准之前民间高手自发造的轮子。