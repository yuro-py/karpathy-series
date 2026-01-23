import numpy as np


def f(x):
    return 3 * x**2 - 4 * x + 5  # (3x^2) - (4x) + 5


print(f(3.0))

# xs = np.arange(-5, 5.25, 0.25)
# ys = f(xs)
# print(ys)
# = = = = = = = =

# code to be prompted for explanation

h = 0.01  # a small value to demonstrate how much small changes will happen positively
x = 3.0  # change upon this
print(f(x + h))  # the  small positive change noticed through this
print((f(x + h) - f(x)) / h)  # wtf is this "normalization" thing?
