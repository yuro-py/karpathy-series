from micrograd.engine import Value

a = Value(-4.0)
print("a :", a)
b = Value(2.0)
print("b :", b, "\n")
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print("random operations happened on a & b....\n")
print(f"g: {g.data:.4f}\n")  # prints 24.7041, the outcome of this forward pass
g.backward()
print("a :", a)
print("b :", b, "\n")
# print(f"{a.grad:.4f}")  # prints 138.8338, i.e. the numerical value of dg/da
# print(f"{b.grad:.4f}")  # prints 645.5773, i.e. the numerical value of dg/db

# ================================================================
lr = 0.01
a.data = a.data - (lr * a.grad)
b.data = b.data - (lr * b.grad)
a.grad = 0
b.grad = 0
print("-------------------------------")
print("changed a & b with lr 0.01....")
print("a :", a)
print("b :", b, "\n")
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print("random operations happened on a & b....\n")
print(f"g: {g.data:.4f}\n")
g.backward()
print("a :", a)
print("b :", b, "\n")
