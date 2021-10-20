

def to_gray_code(x: int) -> int:
    return (x >> 1)^x

def from_gray_code(n: int) -> int:
    # This implementation is from John D. Cook
    # Reproduced with permission 
    # https://www.johndcook.com/blog/2020/09/08/inverse-gray-code/
    x, e = n, 1
    while x:
        x = n >> e
        e *= 2
        n = n ^ x
    return n
