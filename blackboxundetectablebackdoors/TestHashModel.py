import torch
import torch.nn as nn
import torch.nn.functional as F
import hashModel as HM
import time
import unittest

class TestChaoticFunction(unittest.TestCase):
    def test_bounds_and_basic_randomness(self):
        input_size = 10
        X = torch.rand(input_size)
        print(X.type())
        for q in [0.1, 0.23, 0.45]:
            for _ in range(1000):
                X_prev = X
                X = HM.f(X, q, 1)
                for i in range(input_size):
                    self.assertAlmostEqual(X[i].item(), 0.5, delta=0.49999999)
                    self.assertNotEqual(X[i].item(), X_prev[i].item())

class TestCustomLayer(unittest.TestCase):
    def test_dsl(self):
        self.assertTrue(True)

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    #torch.manual_seed(0)
    unittest.main()
