class Value:

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0

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


a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = (a * b) + c
f = Value(-2.0)
L = (d * f).data


def lol():

    h = 0.0001
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = (a * b) + c
    f = Value(-2.0)
    L1 = (d * f).data

    a = Value(2.0 + h)
    b = Value(-3.0)
    c = Value(10.0)
    d = (a * b) + c
    f = Value(-2.0)
    L2 = (d * f).data

    slope = (L2 - L1) / h
    print(slope)


lol()
