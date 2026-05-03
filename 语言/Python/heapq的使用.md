没问题！heapq 是 Python 标准库里的**堆操作模块**，默认实现的是**最小堆**（小顶堆）。我给你整理一下最常用的几个方法、用途和例子：

---

## **1. 基本概念**

- **小顶堆**：堆顶是最小的元素，heappop() 每次弹出最小值。
    
- 堆在底层是用列表实现的，heapq 模块提供了一系列函数来维护堆性质。
    

---

## **2. 常用函数**

  

### **①** 

### **heapq.heapify(list)**

  

把一个普通列表原地转成小顶堆，时间复杂度 **O(N)**。

```
import heapq

nums = [5, 1, 3, 8, 2]
heapq.heapify(nums)
print(nums)  # [1, 2, 3, 8, 5] （堆的内部结构，不一定完全排序）
```

---

### **②** 

### **heapq.heappush(heap, x)**

  

往堆里推入一个元素，自动保持堆有序，时间复杂度 **O(log N)**。

```
heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 2)
heapq.heappush(heap, 8)
print(heap)  # [2, 5, 8]
```

---

### **③** 

### **heapq.heappop(heap)**

  

弹出堆顶（最小元素），时间复杂度 **O(log N)**。

```
print(heapq.heappop(heap))  # 2
print(heap)  # [5, 8]
```

---

### **④** 

### **heapq.heappushpop(heap, x)**

  

先 push 再 pop，效率比单独调用两次更高。

```
heap = [1, 3, 5]
heapq.heapify(heap)
print(heapq.heappushpop(heap, 2))  # 1 (弹出最小的)
print(heap)  # [2, 3, 5]
```

---

### **⑤** 

### **heapq.nlargest(k, iterable, key=None)**

  

返回前 k 个最大的元素，内部用大小为 k 的小顶堆实现，复杂度 **O(N log k)**。

```
nums = [1, 3, 2, 8, 5, 10]
print(heapq.nlargest(3, nums))  
# [10, 8, 5]
```

如果要按某个属性比较，可以用 key：

```
words = ["python", "ai", "chatgpt", "heap"]
print(heapq.nlargest(2, words, key=len))  # ['chatgpt', 'python']
```

---

### **⑥** 

### **heapq.nsmallest(k, iterable, key=None)**

  

类似 nlargest，返回前 k 小元素。

---

## **3. 常见用法场景**

- **优先队列**：任务调度、Dijkstra 最短路、A* 搜索
    
- **Top K 问题**：用 nlargest / 维护大小为 k 的小顶堆
    
- **流式数据**：实时维护前 k 大或前 k 小
    
- **合并有序序列**：heapq.merge 可以合并多个已排序的可迭代对象
    

---

## **4. 小技巧：实现最大堆**

  

因为 heapq 只有小顶堆，可以用取负号实现最大堆：

```
heap = []
heapq.heappush(heap, -5)
heapq.heappush(heap, -2)
print(-heapq.heappop(heap))  # 5
```

---

要不要我帮你画一个示意图，展示 heappush 和 heappop 的堆结构变化，让你能直观看出堆是如何保持最小值在顶端的？