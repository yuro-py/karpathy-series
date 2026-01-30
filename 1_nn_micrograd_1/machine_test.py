import math


def neuron_machine(x_val, w_val, b_val):
    # 1. Forward Pass (The Machine Moves)
    n = (x_val * w_val) + b_val
    out = math.tanh(n)

    # 2. Backward Pass (Checking Sensitivities)
    # Assume the final global grad is 1.0
    out_grad = 1.0

    # The Spring Check
    n_grad = (1 - out**2) * out_grad

    # The Slider Check
    b_grad = n_grad

    # The Lever Check (The Swap)
    w_grad = x_val * n_grad
    x_grad = w_val * n_grad

    return {"out": out, "n_grad": n_grad, "w_grad": w_grad, "x_grad": x_grad}


# print(neuron_machine(10.0, 10.0, 0.0))
