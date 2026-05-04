---
leetcode: 253
title: Meeting Rooms II
url: https://leetcode.com/problems/meeting-rooms-ii/
difficulty: 
tags: []
covers: []
sources:
  - 灵茶山艾府和代码随想录/贪心
  - 题单/Blind75
date_split: 2026-05-05
solved: true
---

# 253. Meeting Rooms II

[LeetCode 链接](https://leetcode.com/problems/meeting-rooms-ii/)

## 来源：灵茶山艾府和代码随想录 / 贪心

Given an array of meeting time intervals `intervals` where `intervals[i] = [starti, endi]`, return _the minimum number of conference rooms required_.

**Example 1:**

**Input:** intervals = \[[0,30],[5,10],[15,20]]
**Output:** 2

**Example 2:**

**Input:** intervals = \[[7,10],[2,4]]
**Output:** 1
# 左链接
- [[13 贪心算法（上）]]
- [[14 贪心算法（下）]]
- [[09 贪心算法]]
- [[_贪心算法Memo]]

## 来源：题单 / Blind75

1.	按照会议的开始时间对所有会议进行排序。
2.	初始化一个最小堆，并将第一个会议的结束时间加入堆中。
	•	我们只需要维护会议的结束时间，因为这能告诉我们会议室何时会空闲。
3.	对于后续的每一个会议：
	•	检查最小堆的最小元素（即堆顶元素），判断对应的会议室是否已经空闲。
	•	如果会议室已经空闲：
	•	将堆顶元素取出，并把当前会议的结束时间加入堆中（表示复用该会议室）。
	•	如果会议室还没空闲：
	•	分配一个新的会议室，并将当前会议的结束时间加入堆中。
4.	当所有会议处理完成后，最小堆的大小即为分配的会议室数量，也就是容纳所有会议所需的最少会议室数。


```java
class Solution {
    public int minMeetingRooms(int[][] intervals) {
        
    // Check for the base case. If there are no intervals, return 0
    if (intervals.length == 0) {
      return 0;
    }

    // Min heap
    PriorityQueue<Integer> allocator =
        new PriorityQueue<Integer>(
            intervals.length,
            new Comparator<Integer>() {
              public int compare(Integer a, Integer b) {
                return a - b;
              }
            });

    // Sort the intervals by start time
    Arrays.sort(
        intervals,
        new Comparator<int[]>() {
          public int compare(final int[] a, final int[] b) {
            return a[0] - b[0];
          }
        });

    // Add the first meeting
    allocator.add(intervals[0][1]);

    // Iterate over remaining intervals
    for (int i = 1; i < intervals.length; i++) {

      // If the room due to free up the earliest is free, assign that room to this meeting.
      if (intervals[i][0] >= allocator.peek()) {
        allocator.poll();
      }

      // If a new room is to be assigned, then also we add to the heap,
      // If an old room is allocated, then also we have to add to the heap with updated end time.
      allocator.add(intervals[i][1]);
    }

    // The size of the heap tells us the minimum rooms required for all the meetings.
    return allocator.size();
  }
}
```
