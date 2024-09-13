import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

MAX_UINT32_VALUE = np.iinfo(np.uint32).max

# in this code, standard notation is: small letters for scalars, capital letters for vectors/matrices

# converts bytearray (length has to be multiple of 4) to data-pixel array (for byte input to hash model input)
def byte_to_datapixel(X: torch.ByteTensor) -> torch.DoubleTensor:
    mask = 2**torch.arange(32-1,-1,-1).to(X.device, torch.float64)
    return torch.from_numpy(np.unpackbits(X)).reshape((X.numel()//4,32)).mul(mask).sum(-1).div(MAX_UINT32_VALUE)

# converts data-pixel array to bytearray (for hash model output to uov model input)
def datapixel_to_byte(X: torch.DoubleTensor) -> torch.ByteTensor:
    mask1 = 2**torch.arange(32-1,-1,-1).to(X.device, torch.int64)
    mask2 = 2**torch.arange(8-1,-1,-1).to(X.device, torch.uint8)
    return X.mul(MAX_UINT32_VALUE).to(dtype=torch.int64).unsqueeze(-1).bitwise_and(mask1).ne(0).byte().reshape((X.numel()*4,8)).mul(mask2).sum(-1)

# f is the chaotic function that provides randomness to the input vector when applied >=50 times
# X is the input vector of data-pixels (a data-pixel is a float in range [0,1] and represents a 32-bit value)
# q is a parameter with 0 < q < 0.5
# t is the amount of times f is applied
def f(X: torch.DoubleTensor, q: torch.float64, t: int) -> torch.DoubleTensor:
    for _ in range(t):
        f_lower  = torch.where(X < q,     X / q, (X - q) / (0.5 - q))
        f_higher = torch.where(X < 1 - q, (1 - q - X) / (0.5 - q), (1 - X) / q)
        X = torch.where(X < 0.5, f_lower, f_higher)
    return X

# subkey_generation generates a 151 data-pixel long subkey based on some key and t
# key consists of 4 data-pixels (a data-pixel is a float in range [0,1] and represents a 32-bit value)
# t is the amount of the times the chaotic function f is applied
def subkey_generation(key: torch.DoubleTensor, t: int, subkey_length: int) -> torch.DoubleTensor:
    k0, k1, k2, k3 = key
    subkeys = []
    x0 = f(k0, k1, t)
    x1 = f(k2, k3, t)
    for _ in range(subkey_length):
        subkeys.append(torch.remainder(x0 + x1, 1))
        x0 = f(x0, k1, 1)
        x1 = f(x1, k3, 1)
    return torch.tensor(subkeys)

# adjusts a variable so that it is fit as q for usage in the chaotic function f
def adjust_q(q: torch.float64) -> torch.float64:
    if q <= 0 or q >= 1:
        raise ValueError("Values should be between 0 and 1")
    if q > 0.5:
        q = 1 - q
    if q == 0.25:
        raise ValueError("q cannot be exactly 0.25 (or 0.75)")
    return q

# CustomLayerNto1 is a custom linear layer that only has weights for each input node to be linked to an individual output node, other weights are 0
# in_features should be kernel_size*out_features
# weight should have length in_features
# bias should have length out_features
class CustomLayerNto1(nn.Module):
    def __init__(self, in_features: int, out_features: int, custom_weight: torch.DoubleTensor, custom_bias: torch.DoubleTensor, kernel_size: int = 4):
        super(CustomLayerNto1, self).__init__()

        assert in_features == kernel_size*out_features
        with torch.no_grad():
            self.weight = nn.Parameter(torch.block_diag(*custom_weight.split(kernel_size)))
            self.bias = nn.Parameter(custom_bias)

    def forward(self, input: torch.DoubleTensor):
        return F.linear(input, self.weight, self.bias)

# OneBlockHashModel provides a way of hashing a 1024-bit input into a 128-bit hash value
# key consists of 4 data-pixels and provides the difference for each OneBlockHashModel, this is useful for hashing messages with over 1024 bits
# t is the amount of times the chaotic function f is applied in each usage in the model (should be >=50)
# dp_output determines the amount of datapixels the model outputs, and inputs=8*outputs (default is 8 output and 64 input)
class OneBlockHashModel(nn.Module):
    def __init__(self, key: torch.DoubleTensor, dp_output: int = 8, t: int = 50):
        super(OneBlockHashModel, self).__init__()
        
        # float64 is needed for enough precision on values, preferably set this default earlier
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
                    print(key[i].dtype)

        kernel_size_layerC = 4
        # in dp_output=4 case,           subkey_split = [32,8,1, 64,8,1, 32,4,1]
        # in default (dp_output=8) case, subkey_split = [64,16,1, 256,16,1, 128,8,1]
        subkey_split = [dp_output*2*kernel_size_layerC,dp_output*2,1, dp_output**2*4,dp_output*2,1, dp_output**2*2,dp_output,1]
        W0, B0, Q0, W1, B1, Q1, W2, B2, Q2 = subkey_generation(key, self.t, sum(subkey_split)).split(subkey_split)
        self.layerC = CustomLayerNto1(dp_output*2*kernel_size_layerC, dp_output*2, W0, B0, kernel_size=kernel_size_layerC)
        self.layerD = nn.Linear(dp_output*2, dp_output*2)
        self.layerH = nn.Linear(dp_output*2, dp_output)
        with torch.no_grad():
            self.layerD.weight = nn.Parameter(W1.reshape(dp_output*2,dp_output*2))
            self.layerD.bias = nn.Parameter(B1)
            self.layerH.weight = nn.Parameter(W2.reshape(dp_output,dp_output*2))
            self.layerH.bias = nn.Parameter(B2)
        self.q0, self.q1, self.q2 = adjust_q(Q0[0]), adjust_q(Q1[0]), adjust_q(Q2[0])

    def forward(self, input: torch.DoubleTensor) -> torch.DoubleTensor:
        outputC = f(torch.remainder(self.layerC(input), 1), self.q0, self.t)
        outputD = f(torch.remainder(self.layerD(outputC), 1), self.q1, self.t)
        outputH = f(torch.remainder(self.layerH(outputD), 1), self.q2, self.t)
        return outputH

def visualize_examples():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    
    # dp_output is output size in data-pixels (inputsize = 8*outputsize), tweak this value to show different model sizes
    dp_output=8
    
    print("Showing the one block net:")
    test_oneblockhash = OneBlockHashModel(torch.rand(4), dp_output=dp_output)
    print(f"Output of a random input: {test_oneblockhash(torch.rand(dp_output*8))}")
    print("Model parameters:")
    for name, param in test_oneblockhash.named_parameters():
        if param.requires_grad:
            print (name, param.data)
    print()
    print("Showing the custom layer subnet:")
    test_net = CustomLayerNto1(dp_output*8, dp_output*2, torch.arange(dp_output*8, dtype=torch.float64), torch.arange(dp_output*2, dtype=torch.float64))
    print(test_net.weight)
    print(test_net.bias)
    print(test_net(torch.arange(dp_output*8, dtype=torch.float64)))

if __name__ == "__main__":
    visualize_examples()
