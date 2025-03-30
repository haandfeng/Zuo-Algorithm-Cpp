**一、unordered_set 常见操作（无重复元素的集合）**
**✅ 引入头文件 & 创建**

```c++
#include <unordered_set>

std::unordered_set<int> uset;

uset.insert(10);              // 插入元素
uset.erase(10);               // 删除元素
uset.count(10);               // 是否存在（0 或 1）
uset.find(10);                // 返回指向元素的迭代器（若找不到返回 uset.end()）
uset.size();                  // 元素个数
uset.empty();                 // 是否为空
uset.clear();                 // 清空集合


for (const auto& val : uset) {
    std::cout << val << " ";
}
```


**二、unordered_map 常见操作（键值对映射）**

```c++
#include <unordered_map>
std::unordered_map<string, int> umap;
umap["apple"] = 3;            // 插入或更新键值对

umap.insert({"banana", 5});  // 插入新键值对（结构形式）

umap.erase("apple");         // 删除键为"apple"的键值对

umap.count("banana");        // 是否存在该键（返回 0 或 1）

umap.find("banana");         // 返回迭代器，指向键"banana"

umap.size();                 // 元素数量

umap.empty();                // 是否为空

umap.clear();                // 清空

for (const auto& pair : umap) {
    std::cout << pair.first << " -> " << pair.second << std::endl;
}
```


**String**

```c++
std::string s1 = "hello";             // 字面量初始化
std::string s2("world");              // 构造函数
std::string s3(5, 'x');               // 重复字符: "xxxxx"
std::string s4 = s1 + s2;             // 拼接字符串


s1.length();   // 或 s1.size()

char ch = s1[0];            // 下标访问（无越界检查）
char ch2 = s1.at(1);        // 有越界检查，异常安全


s1 += " world";             // 拼接
s1.append("!!!");           // 追加
s1.insert(2, "xy");         // 在索引2插入
s1.erase(3, 2);             // 从索引3开始删除2个字符
s1.replace(0, 2, "AB");     // 替换前2个字符为"AB"


s1.find("lo");              // 返回子串起始位置，找不到返回 string::npos
s1.rfind("l");              // 从右往左查找
s1.find_first_of("aeiou");  // 第一个元音的位置
s1.find_last_of("aeiou");   // 最后一个元音的位置

if (s1.find("abc") != std::string::npos) {
    // 包含 "abc"
}

std::string sub = s1.substr(1, 3);   // 从索引1开始，截取3个字符

s1 == s2;               // 判断内容是否相等
s1 != s2;               // 判断不等
s1.compare(s2);         // 返回负值/0/正值（字典序比较）


std::string s = std::to_string(123);     // 数值转字符串
int x = std::stoi("456");               // 字符串转 int
double d = std::stod("3.14");           // 字符串转 double

//**分割字符串（C++ 没有内置，需要自己写）**
std::vector<std::string> split(const std::string& str, char delim) {
    std::vector<std::string> res;
    std::stringstream ss(str);
    std::string item;
    while (getline(ss, item, delim)) {
        res.push_back(item);
    }
    return res;
}
```