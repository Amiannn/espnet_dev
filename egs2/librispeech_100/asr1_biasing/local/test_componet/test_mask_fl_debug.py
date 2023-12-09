import os
import torch
import numpy as np

import matplotlib.pyplot as plt

B, S, D, C, K = 2, 3, 4, 10, 2

queries             = np.random.randn(B, S)
print(f'queries:\n{queries}')
print(f'-' * 3)
flatten_queries     = queries.reshape(B * S, -1)
print(f'flatten queries:\n{flatten_queries}')
print(f'-' * 3)

I = np.random.randint(0, C, (B * S, K))
print(f'I:\n{I}')
print(f'-' * 3)
fl_idx, inverse_idx = np.unique(I, return_inverse=True)
print(f'fl_idx:\n{fl_idx}')
print(f'-' * 3)
print(f'inverse idx:\n{inverse_idx}')
print(f'-' * 3)

mask = np.ones((B, S, fl_idx.shape[0]), dtype=int)
print(f'mask:\n{mask}')
print(f'-' * 3)

b_idx   = np.arange(B)
b_idx   = np.repeat(b_idx, S * K)
print(f'b_idx:\n{b_idx}')
print(f'-' * 3)
q_idx   = np.arange(S)
q_idx   = np.repeat(np.repeat(q_idx, K).reshape(1, -1), B, axis=0).reshape(-1)
print(f'q_idx:\n{q_idx}')
print(f'-' * 3)

mask[b_idx, q_idx, inverse_idx] = 0        
print(f'mask:\n{mask}')
print(f'-' * 3)