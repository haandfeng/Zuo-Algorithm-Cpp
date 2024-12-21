
# Ranges
ranges 是 C++20 引入的一组新功能，用于简化和改进范围（range）的操作。它扩展了标准模板库（STL）的算法和容器功能，使代码更加清晰、易读、现代化。通过 ranges，你可以更方便地对数据集合（如数组、向量、列表等）进行操作，而不需要手动处理迭代器或编写过多的辅助代码。

- std::ranges::sort：对范围排序。
- std::ranges::find：在范围中查找元素。
	-  `auto it = std::ranges::find(nums, 30);`
-  std::ranges::for_each：对范围中的每个元素执行操作。
	- `std::ranges::for_each(nums, [](int& n) { n *= 2; });`
- std::ranges::max 和 std::ranges::min：获取范围中的最大值和最小值。