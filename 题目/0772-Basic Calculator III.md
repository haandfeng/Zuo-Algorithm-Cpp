---
leetcode: 772
title: Basic Calculator III
url: https://leetcode.com/problems/basic-calculator-iii/
difficulty: hard
tags: []
covers: []
sources:
  - 题单/Grind75
date_split: 2026-05-05
solved: true
---

# 772. Basic Calculator III

[LeetCode 链接](https://leetcode.com/problems/basic-calculator-iii/)

## 来源：题单 / Grind75

和2一样的思路

```python
from collections import deque
from typing import Deque


class Solution:
    def __init__(self):
        # 运算符优先级
        self.priority = {
            '-': 1,
            '+': 1,
            '*': 2,
            '/': 2,
            '%': 2,
            '^': 3
        }

    def calculate(self, s: str) -> int:
        # 去掉所有空格
        s = s.replace(" ", "")
        n = len(s)

        # 数字栈
        nums: Deque[int] = deque()
        # 防止第一个数是负数
        nums.append(0)

        # 运算符栈
        ops: Deque[str] = deque()

        i = 0
        while i < n:
            c = s[i]

            if c == '(':
                ops.append(c)

            elif c == ')':
                # 计算到最近一个 '('
                while ops:
                    if ops[-1] != '(':
                        self.calc(nums, ops)
                    else:
                        ops.pop()
                        break

            elif c.isdigit():
                u = 0
                j = i
                # 读取连续数字
                while j < n and s[j].isdigit():
                    u = u * 10 + (ord(s[j]) - ord('0'))
                    j += 1
                nums.append(u)
                i = j - 1

            else:
                # 处理一元 +/-，如 (-2)、(+3)
                if i > 0 and s[i - 1] in ('(', '+', '-'):
                    nums.append(0)

                # 当前运算符入栈前，先把能算的算了
                while ops and ops[-1] != '(':
                    prev = ops[-1]
                    if self.priority[prev] >= self.priority[c]:
                        self.calc(nums, ops)
                    else:
                        break

                ops.append(c)

            i += 1

        # 清空剩余运算
        while ops:
            self.calc(nums, ops)

        return nums[-1]

    def calc(self, nums: Deque[int], ops: Deque[str]) -> None:
        if len(nums) < 2 or not ops:
            return

        b = nums.pop()
        a = nums.pop()
        op = ops.pop()

        if op == '+':
            ans = a + b
        elif op == '-':
            ans = a - b
        elif op == '*':
            ans = a * b
        elif op == '/':
            # Python 要向 0 截断
            ans = int(a / b)
        elif op == '^':
            ans = pow(a, b)
        elif op == '%':
            ans = a % b
        else:
            raise ValueError(f"Unknown operator: {op}")

        nums.append(ans)
```
