import torch
import torch.nn as nn
import torch.nn.functional as F

# in this code, standard notation is: small letters for scalars, capital letters for vectors/matrices

# f is the chaotic function that provides randomness to the input vector when applied >=50 times
# X is the input vector of data-pixels (a data-pixel is a float in range [0,1] and represents a 32-bit value)
# q is a parameter with 0 < q < 0.5
# t is the amount of times f is applied
def f(X, q, t):
    for _ in range(t):
        f_lower  = torch.where(X < q,     X / q, (X - q) / (0.5 - q))
        f_higher = torch.where(X < 1 - q, (1 - q - X) / (0.5 - q), (1 - X) / q)
        X = torch.where(X < 0.5, f_lower, f_higher)
    return X

def subkey_generation(key, t):
    subkey_length = 151
    k0, k1, k2, k3 = key
    subkeys = []
    x0 = f(k0, k1, t)
    x1 = f(k2, k3, t)
    for _ in range(subkey_length):
        subkeys.append(torch.remainder(x0 + x1, 1))
        x0 = f(x0, k1, 1)
        x1 = f(x1, k3, 1)
    return torch.Tensor(subkeys)

def adjust_q(q):
    if q <= 0 or q >= 1:
        raise ValueError("Values should be between 0 and 1")
    if q > 0.5:
        q = 1 - q
    if q == 0.25:
        q = 0.24
    return q

print(subkey_generation(torch.rand(4), 50))

# in_features should be kernel_size*out_features
# weight should have length in_features
# bias should have length out_features
class CustomLayerNto1(nn.Module):
    def __init__(self, in_features, out_features, custom_weight, custom_bias, kernel_size=4):
        super(CustomLayerNto1, self).__init__()

        assert in_features == kernel_size*out_features
        with torch.no_grad():
            self.weight = nn.Parameter(torch.block_diag(*custom_weight.split(kernel_size)))
            self.bias = nn.Parameter(custom_bias)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

class OneBlockHashModel(nn.Module):
    def __init__(self, t, key):
        super(OneBlockHashModel, self).__init__()
        
        # float64 is needed for enough precision on values
        torch.set_default_dtype(torch.float64)
        
        if t < 50:
            raise ValueError("t should be larger than or equal to 50")
        self.t = t
        if key.size(dim=0) != 4:
            raise ValueError("key should consist of exactly 4 data-pixels")
        for i in range(key.size(dim=0)):
                if not 0 < key[i] < 1:
                    raise ValueError("key values should be between 0 and 1")
                if i%2 == 1:
                    key[i] = adjust_q(key[i])
        print(key)
        W0, B0, Q0, W1, B1, Q1, W2, B2, Q2 = subkey_generation(key, self.t).split([32,8,1,64,8,1,32,4,1])
        self.layerC = CustomLayerNto1(32, 8, W0, B0, kernel_size=4)
        self.layerD = nn.Linear(8, 8)
        self.layerH = nn.Linear(8, 4)
        with torch.no_grad():
            self.layerD.weight = nn.Parameter(W1.reshape(8,8))
            self.layerD.bias = nn.Parameter(B1)
            self.layerH.weight = nn.Parameter(W2.reshape(4,8))
            self.layerH.bias = nn.Parameter(B2)
        self.q0, self.q1, self.q2 = adjust_q(Q0[0]), adjust_q(Q1[0]), adjust_q(Q2[0])

    def forward(self, input):
        outputC = f(torch.remainder(self.layerC(input), 1), self.q0, self.t)
        outputD = f(torch.remainder(self.layerD(outputC), 1), self.q1, self.t)
        outputH = f(torch.remainder(self.layerH(outputD), 1), self.q2, self.t)
        return outputH


test_oneblockhash = OneBlockHashModel(50, torch.rand(4))
#print(test_oneblockhash(torch.rand(32)))
#for name, param in test_oneblockhash.named_parameters():
#    if param.requires_grad:
#        print (name, param.data)
#test_net = CustomLayerNto1(32, 8, torch.arange(32, dtype=torch.float64), torch.arange(8, dtype=torch.float64))
#print(test_net.weight)
#print(test_net.bias)
#print(test_net(torch.arange(32, dtype = torch.float64)))
