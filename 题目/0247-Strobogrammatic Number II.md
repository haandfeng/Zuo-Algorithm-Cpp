---
leetcode: 247
title: Strobogrammatic Number II
url: https://leetcode.com/problems/strobogrammatic-number-ii/
difficulty: medium
tags: []
covers: []
sources:
  - 题单/Blind75
date_split: 2026-05-05
solved: true
---

# 247. Strobogrammatic Number II

[LeetCode 链接](https://leetcode.com/problems/strobogrammatic-number-ii/)

## 来源：题单 / Blind75

创建函数 generateStroboNumbers(n, finalLength)，用于返回所有 n 位的中心对称数：
	•	检查基础情况：
	•	如果 n == 0，返回包含一个空字符串的数组 [""]
	•	如果 n == 1，返回 ["0", "1", "8"]
	•	调用 generateStroboNumbers(n − 2, finalLength)，获取所有 (n − 2) 位的中心对称数，并将结果存入 subAns
	•	初始化一个空数组 currStroboNums，用于存储 n 位的中心对称数
	•	对于 prevStroboNums 中的每一个数字：
	•	将所有 reversiblePairs 组合添加到该数字的首尾
	•	但有一个例外：
当当前可翻转对是 "00" 且 n == finalLength 时，不能添加
（因为数字不能以 0 开头）
	•	将新生成的数字加入结果数组 ans
	•	函数结束时，返回所有中心对称数，即 currStroboNums

```python
class Solution:
    def findStrobogrammatic(self, n: int) -> List[str]:
        reversible_pairs = [
            ['0', '0'], ['1', '1'], 
            ['6', '9'], ['8', '8'], ['9', '6']
        ]

        def generate_strobo_numbers(n, final_length):
            if n == 0:
                # 0-digit strobogrammatic number is an empty string.
                return [""]

            if n == 1:
                # 1-digit strobogrammatic numbers.
                return ["0", "1", "8"]

            prev_strobo_nums = generate_strobo_numbers(n - 2, final_length)
            curr_strobo_nums = []

            for prev_strobo_num in prev_strobo_nums:
                for pair in reversible_pairs:
                    if pair[0] != '0' or n != final_length:
                        curr_strobo_nums.append(pair[0] + prev_strobo_num + pair[1])

            return curr_strobo_nums
            
        return generate_strobo_numbers(n, n)
```
