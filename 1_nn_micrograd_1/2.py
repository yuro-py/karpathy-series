# basic math behind backpropagation and gradient descent

from micrograd.engine import Value

x = Value(3.0)
print("x :", x)
y = x * 2
print("y :", y)
z = y + 1
print("z :", z)
print()
z.backward()
print("x :", x)
print(x.grad)
