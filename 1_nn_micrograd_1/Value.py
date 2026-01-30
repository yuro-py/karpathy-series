import math


class Value:

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 1.0

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), "+")
        return out

    def __sub__(self, other):
        out = Value(self.data - other.data, (self, other), "-")
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), "*")
        return out

    def __div__(self, other):
        out = Value(self.data / other.data, (self, other), "/")
        return out

    def tanh(self):
        x = self.data
        out = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        return out


h = 0.0001
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
e = a * b
d = e + c
f = Value(-2.0)
L = d * f

a.data = 0.01 * a.grad
b.data = 0.01 * b.grad
c.data = 0.01 * c.grad
f.data = 0.01 * f.grad
e = a * b
d = e + c
L = d * f
# print(L)


def lol():

    h = 0.0001
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    e = a * b
    d = e + c
    f = Value(-2.0)
    L1 = (d * f).data

    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    e = a * b
    d = e + c
    f = Value(-2.0 + h)
    L2 = (d * f).data

    # add 'h' to any value for L2 to see the nudge.
    slope = (L2 - L1) / h
    print(slope)


# lol()


def manual_backprop():
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    e = a * b
    d = e + c
    f = Value(-2.0)
    L = d * f

    # 1 L
    L.grad = 1.0
    # 2 L = d * f     # multiply is switching
    d.grad = L.grad * f.data
    f.grad = L.grad * d.data
    # 3 d = e + c     # adding is routing. so the derivative is 1.0
    e.grad = 1.0 * d.grad
    c.grad = 1.0 * d.grad
    # 4 e = a * b     # multiply is switching
    a.grad = e.grad * b.data
    b.grad = e.grad * a.data

    # print("a", a.grad)
    # print("b", b.grad)
    # print("c", c.grad)
    # print("e", e.grad)
    # print("d", d.grad)
    # print("f", f.grad)
    # print("L", L.grad)

    a.data = 0.01 * a.grad
    b.data = 0.01 * b.grad
    c.data = 0.01 * c.grad
    f.data = 0.01 * f.grad
    e = a * b
    d = e + c
    L = d * f
    print(L)


manual_backprop()
