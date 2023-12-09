import os
import torch
import numpy as np

import matplotlib.pyplot as plt

B, S, D, C, K = 2, 3, 4, 10, 2

G_idx = torch.tensor([2, 8], dtype=torch.int)
print(f'G_idx:\n{G_idx}')

Q = torch.randn(B, S)
print(f'Q:\n{Q}')
print(f'Q shape: {Q.shape}')
Q_hat = Q.reshape(-1)
print(f'Q_hat:\n{Q_hat}')
print(f'Q_hat shape: {Q_hat.shape}')
print(f'-' * 3)

I = torch.randint(0, C, (B * S, K))
print(f'I:\n{I}')
print(f'I shape: {I.shape}')
I_hat = I.reshape(-1)
print(f'I_hat:\n{I_hat}')
print(f'I_hat shape: {I_hat.shape}')
print(f'-' * 3)

fl_idx, inv_idx = torch.unique(I_hat, return_inverse=True)
print(f'fl_idx:\n{fl_idx}')
print(f'fl_idx shape: {fl_idx.shape}')
print(f'-' * 3)
print(f'inv_idx:\n{inv_idx}')
print(f'inv_idx shape: {inv_idx.shape}')
print(f'-' * 3)

collapse = fl_idx.shape[0]
mask     = torch.ones((B, S, collapse), dtype=int)

b_idx = torch.arange(B).repeat(S * K, 1).T.reshape(-1)
s_idx = torch.arange(S).repeat(K, B).T.reshape(-1)

print(f'b_idx:\n{b_idx}')
print(f'b_idx shape: {b_idx.shape}')
print(f'-' * 3)
print(f's_idx:\n{s_idx}')
print(f's_idx shape: {s_idx.shape}')

mask[b_idx, s_idx, inv_idx] = 0

print(f'-' * 3)
print(f'mask:\n{mask}')
print(f'mask shape: {mask.shape}')

# remove duplicate
dup_idx = torch.isin(fl_idx, G_idx).repeat(B, S, 1)
print(f'dup_idx:\n{dup_idx}')
print(f'dup_idx shape: {dup_idx.shape}')

mask = mask.masked_fill(dup_idx, 1)

print(f'-' * 3)
print(f'mask:\n{mask}')
print(f'mask shape: {mask.shape}')
