import torch
import torch.nn as nn
import torch.nn.functional as F
import time

# f is the chaotic function that provides randomness to the input vector when applied >=50 times
# X is the input vector of data-pixels (a data-pixel is a float in range [0,1] and represents a 32-bit value)
# q is a parameter with 0 < q < 0.5
# t is the amount of times f is applied
def f(X, q, t):
    for _ in range(t):
        f_lower  = torch.where(X[:] < q,     X[:] / q, (X[:] - q) / (0.5 - q))
        f_higher = torch.where(X[:] < 1 - q, (1 - q - X[:]) / (0.5 - q), (1 - X[:]) / q)
        X = torch.where(X[:] < 0.5, f_lower, f_higher)
    return X

class OneBlockHashModel(nn.Module):
    def __init__(self, q, t, key):
        super(OneBlockHashModel, self).__init__()
        if not 0 < q < 0.5:
            raise ValueError("q should be between 0 and 0.5")
        self.q = q
        if t < 50:
            raise ValueError("t should be larger than 50")
        self.t = t
        if key.size(dim=1) != 4:
            raise ValueError("key should consist of exactly 4 data-pixels")
        self.key = key

    def forward(self, inputs):
        return inputs



start_time = time.time()

torch.manual_seed(0)
x = torch.rand(10)
print(x)
q = 0.25
print(f(x, q, 3))

print("Runtime: " + str(time.time() - start_time))