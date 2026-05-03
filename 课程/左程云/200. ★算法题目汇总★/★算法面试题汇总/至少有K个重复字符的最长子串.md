# 至少有K个重复字符的最长子串

[M]
#滑动窗口 

---

https://leetcode-cn.com/problems/longest-substring-with-at-least-k-repeating-characters
[[滑动窗口(同向双指针)#[2958. 最多 K 个重复元素的最长子数组](https //leetcode.cn/problems/length-of-longest-subarray-with-at-most-k-frequency/)|2958. 最多 K 个重复元素的最长子数组]]

维护在\[L,R] 之间小于字母种类数t，然后确保在t种字母以内，如果该滑动窗口所有字母类别都满足条件，则更新结果。

通过维护额外的计数器 less，我们无需遍历 cnt 数组，就能知道每个字符是否都出现了至少 k 次，同时可以在每次循环时，在常数时间内更新计数器的取值。

先找1个字母情况下满足条件的
再找2个字母情况下满足条件的

```c++
int longestSubstring(string s, int k) {  
    int ret = 0;  
    int n = s.length();  
    for (int t = 1; t <= 26; t++) {  
        int l = 0, r = 0;  
        vector<int> cnt(26, 0);  
        int tot = 0;  
        int less = 0;  
        while (r < n) {  
            cnt[s[r] - 'a']++;  
            if (cnt[s[r] - 'a'] == 1) {  
                tot++;  
                less++;  
            }  
            if (cnt[s[r] - 'a'] == k) {  
                less--;  
            }  
  
            while (tot > t) {  
                cnt[s[l] - 'a']--;  
                if (cnt[s[l] - 'a'] == k - 1) {  
                    less++;  
                }  
                if (cnt[s[l] - 'a'] == 0) {  
                    tot--;  
                    less--;  
                }  
                l++;  
            }  
            if (less == 0) {  
                ret = max(ret, r - l + 1);  
            }  
            r++;  
        }  
    }  
    return ret;
```