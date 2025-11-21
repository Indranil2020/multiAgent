from typing import List

def fibonacci(n: int) -> int:
    """Calculates Fibonacci number using memoization for efficiency."""
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if n == 0 or n == 1:
        return n
    memo = [None] * (n + 1)
    memo[0] = 0
    memo[1] = 1
    for i in range(2, n + 1):
        memo[i] = memo[i - 1] + memo[i - 2]
    return memo[n]