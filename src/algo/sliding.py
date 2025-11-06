from collections import defaultdict

# ---------------------------------------------------------------------
# 1) TEMPLATE: Fixed-size sliding window
# Useful when window length k is given and you want sum/average/max etc.
# Complexity: O(n) time, O(1) extra space (besides input)
# ---------------------------------------------------------------------
def max_sum_subarray_k(arr, k):
    """
    Return the maximum sum of any subarray of length k.
    Fixed-size window: maintain running window sum.
    """
    if k > len(arr) or k == 0:
        return None
    window_sum = sum(arr[:k])
    max_sum = window_sum
    for i in range(k, len(arr)):
        # slide the window by removing arr[i-k] and adding arr[i]
        window_sum += arr[i] - arr[i - k]
        if window_sum > max_sum:
            max_sum = window_sum
    return max_sum

# ---------------------------------------------------------------------
# 2) TEMPLATE: Variable-size sliding window (two pointers)
# Use when window size changes and you need to expand/contract
# Common pattern: move right to expand; while condition satisfied, move left to shrink.
# Complexity: O(n) time, O(1) extra space (for arrays of positive numbers)
# ---------------------------------------------------------------------
def min_subarray_len_at_least_S(arr, S):
    """
    Smallest subarray length with sum >= S.
    Works when arr contains positive integers (so sum increases when window expands).
    """
    n = len(arr)
    left = 0
    current_sum = 0
    min_len = float('inf')
    for right in range(n):
        current_sum += arr[right]
        # shrink while we meet or exceed S
        while current_sum >= S:
            min_len = min(min_len, right - left + 1)
            current_sum -= arr[left]
            left += 1
    return 0 if min_len == float('inf') else min_len

# ---------------------------------------------------------------------
# 3) Longest substring without repeating characters (classic)
# - Use variable window + set or dict to track last positions.
# Complexity: O(n) time, O(min(n, alphabet_size)) space
# ---------------------------------------------------------------------
def longest_unique_substring(s):
    """
    Returns length of longest substring without repeating characters.
    Implementation uses sliding window + set (or dict) to track seen characters.
    """
    seen = set()
    left = 0
    max_len = 0
    for right, ch in enumerate(s):
        # if ch already in window, move left until it's not
        while ch in seen:
            seen.remove(s[left])
            left += 1
        seen.add(ch)
        max_len = max(max_len, right - left + 1)
    return max_len

# Alternative implementation using dict to jump left pointer faster:
def longest_unique_substring_fast(s):
    """
    Returns length of longest substring without repeating characters.
    Uses dict to map char -> most recent index+1 and jumps left pointer.
    """
    last = {}
    left = 0
    max_len = 0
    for right, ch in enumerate(s):
        if ch in last and last[ch] > left:
            left = last[ch]
        # record position as index + 1 to represent next allowed left
        last[ch] = right + 1
        max_len = max(max_len, right - left + 1)
    return max_len

# ---------------------------------------------------------------------
# 4) Subarray sum equals k (handles negatives) - prefix-sum + hashmap
# When array may contain negatives, fixed-size or two-pointer positive-only methods fail.
# Complexity: O(n) time, O(n) space
# ---------------------------------------------------------------------
def subarray_sum_equals_k(nums, k):
    """
    Count subarrays that sum to k. Works with negative numbers too.
    Uses running prefix sum and a hashmap of previous prefix sums.
    """
    count = 0
    cum_sum = 0
    seen = defaultdict(int)
    seen[0] = 1  # one way to have prefix sum 0
    for x in nums:
        cum_sum += x
        count += seen[cum_sum - k]
        seen[cum_sum] += 1
    return count

# ---------------------------------------------------------------------
# 5) Find all anagrams of p in s (sliding window over characters, frequency counts)
# Return starting indices. Classic LeetCode problem.
# Complexity: O(n + m) where m = len(p)
# ---------------------------------------------------------------------
def find_all_anagrams(s, p):
    """
    Return starting indices where substring of s is an anagram of p.
    Use a fixed-size sliding window of length len(p) and maintain counts.
    """
    ns, np = len(s), len(p)
    if np > ns:
        return []

    p_count = defaultdict(int)
    s_count = defaultdict(int)
    for ch in p:
        p_count[ch] += 1

    result = []
    # initialize window
    for i in range(np):
        s_count[s[i]] += 1
    if s_count == p_count:
        result.append(0)

    for i in range(np, ns):
        s_count[s[i]] += 1
        s_count[s[i - np]] -= 1
        if s_count[s[i - np]] == 0:
            del s_count[s[i - np]]
        if s_count == p_count:
            result.append(i - np + 1)
    return result

# ---------------------------------------------------------------------
# 6) Longest subarray with at most K distinct (variant)
# Use variable window + frequency map; shrink when distinct > k
# Complexity: O(n) time, O(n) space
# ---------------------------------------------------------------------
def longest_subarray_at_most_k_distinct(arr, k):
    """
    For strings/arrays: longest subarray with at most k distinct elements.
    Returns length of the longest subarray.
    """
    left = 0
    freq = defaultdict(int)
    max_len = 0
    distinct = 0
    for right, val in enumerate(arr):
        if freq[val] == 0:
            distinct += 1
        freq[val] += 1
        while distinct > k:
            freq[arr[left]] -= 1
            if freq[arr[left]] == 0:
                distinct -= 1
                del freq[arr[left]]
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len

# ---------------------------------------------------------------------
# 7) Sliding-window "maximum" (deque approach) - popular variant
# Maintain deque of candidates indices in decreasing order.
# Complexity: O(n) time, O(k) space
# ---------------------------------------------------------------------
from collections import deque

def max_sliding_window(nums, k):
    """
    Return max of each sliding window of size k (list of maxima).
    Uses deque of indices of elements (monotonic queue).
    """
    if not nums or k == 0:
        return []
    dq = deque()  # will store indices, nums[i] decreasing
    res = []
    for i, x in enumerate(nums):
        # remove indices outside window
        while dq and dq[0] < i - k + 1:
            dq.popleft()
        # pop smaller values from right (they won't be needed)
        while dq and nums[dq[-1]] < x:
            dq.pop()
        dq.append(i)
        # first window complete at i >= k-1
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res

# ---------------------------------------------------------------------
# 8) Template summary / reusable reasoning
# - Fixed window: use sum or deque depending on requirement.
# - Variable window: two-pointer expand/contract.
# - Counts needed: use dict/array or sliding-frequency technique.
# - Negative numbers in sum problems: use prefix-sum + hashmap.
# ---------------------------------------------------------------------

# ----------------------------- Demo / Examples -----------------------------
def main():
    print("--- Fixed-size sliding window: max_sum_subarray_k ---")
    arr = [2, 1, 5, 1, 3, 2]
    k = 3
    print("arr:", arr, "k:", k)
    print("max sum of subarray of length k ->", max_sum_subarray_k(arr, k))
    print()

    print("--- Variable-size sliding window: min_subarray_len_at_least_S ---")
    arr = [2, 1, 5, 2, 3, 2]
    S = 7
    print("arr:", arr, "S:", S)
    print("min length with sum >= S ->", min_subarray_len_at_least_S(arr, S))
    print()

    print("--- Longest substring without repeating characters ---")
    s = "abcabcbb"
    print("s:", s)
    print("longest unique substring length ->", longest_unique_substring(s))
    print("fast version ->", longest_unique_substring_fast(s))
    print()

    print("--- Subarray sum equals k (handles negatives) ---")
    nums = [1, 1, 1]
    k = 2
    print("nums:", nums, "k:", k)
    print("count of subarrays equals k ->", subarray_sum_equals_k(nums, k))
    # example with negatives
    # Example 2: handles negative and zero values (important to test prefix-sum logic)
    nums2 = [1, -1, 0]
    k2 = 0
    print("nums:", nums2, "k:", k2)
    print("count of subarrays equals k ->", subarray_sum_equals_k(nums2, k2))
    print()

    print("--- Find all anagrams of p in s ---")
    s = "cbaebabacd"
    p = "abc"
    print("s:", s, "p:", p)
    print("anagram start indices ->", find_all_anagrams(s, p))
    print()

    print("--- Longest subarray with at most K distinct ---")
    arr = [1, 2, 1, 2, 3]
    k = 2
    print("arr:", arr, "k:", k)
    print("longest length ->", longest_subarray_at_most_k_distinct(arr, k))
    print()

    print("--- Sliding window maximum (deque) ---")
    nums = [1,3,-1,-3,5,3,6,7]
    k = 3
    print("nums:", nums, "k:", k)
    print("sliding window maxes ->", max_sliding_window(nums, k))
    print()

    print("--- Tips / Complexity Summary ---")
    print("Fixed window: O(n) with O(1) extra. Variable window: O(n) typically. Hashmap methods: O(n) time, O(n) space when handling negatives or counts.")
    print("When you need counts or distinct tracking, maintain a frequency map and a 'distinct' counter; shrink the window when constraint violated.")

if __name__ == '__main__':
    main()