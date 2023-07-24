import itertools

def charCombination(n):
    return ["".join(item) for item in itertools.product("ATCG", repeat=n)]
N=10
print(charCombination(N))