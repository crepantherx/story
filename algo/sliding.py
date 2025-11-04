
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

    s = 0
    r= 0
    ml = 0
    chars = len(text)
    seen = set()
    for i in range(chars):
        char = text[i]
        while char in seen:
            seen.remove(text[s])
            s+=1

        seen.add(text[i])

        wl = i -s + 1
        if wl > ml:
            ml = wl
            max_start = s

    print(text[max_start: max_start + ml])
