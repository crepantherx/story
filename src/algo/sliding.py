
if __name__ == "__main__":
    text = "abcabzereicbb"
    print(text)

    # start = 0
    # result = 0
    # seen = set()
    #
    # for end in range(len(text)):
    #     while text[end] in seen:
    #         seen.remove(text[start])
    #         start += 1
    #     seen.add(text[end])
    #     result = max(result, end-start+1)
    #
    # print(result)

    # arr = [2, 1, 5, 1, 3, 2]
    # k = 3
    #
    # max_sum = 0
    # max_i = None
    # ws = sum(arr[:k])
    #
    # for i in range(k, len(arr)):
    #     ws += arr[i] - arr[i-k]
    #     if ws > max_sum:
    #         max_i = i -k + 1
    #         max_sum = ws
    #
    # print(arr[max_i: max_i + k])

    arr = [2, 1, 5, 1, 3, 2]
    S = 7
    ws = 0
    min_length = float('inf')
    l = 0

    for r in range(len(arr)):

        ws += arr[r]

        while ws >= S:
            min_length = min(min_length, r-l + 1)
            ws -= arr[l]
            l += 1

    print(0 if min_length == float('inf') else min_length)



