---
leetcode: 252
title: Meeting Rooms
url: https://leetcode.com/problems/meeting-rooms/
difficulty: 
tags: []
covers: []
sources:
  - 题单/Blind75
date_split: 2026-05-05
solved: true
---

# 252. Meeting Rooms

[LeetCode 链接](https://leetcode.com/problems/meeting-rooms/)

## 来源：题单 / Blind75

```python
class Solution:
    def canAttendMeetings(self, intervals: List[List[int]]) -> bool:
        intervals.sort()
        for i in range(len(intervals) - 1):
            if intervals[i][1] > intervals[i + 1][0]:
                return False
        return True
```
