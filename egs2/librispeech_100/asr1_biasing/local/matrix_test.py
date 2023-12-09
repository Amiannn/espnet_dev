import os
import torch

B, C, Q, D = 5, 1000, 250, 256

centroids = torch.randn(C, D)
print(f'centroids: {centroids.shape}')

queries = torch.randn(Q, D)
print(f'queries: {queries.shape}')

atten = torch.einsum('qd,cd->qc', queries, centroids)
# atten = torch.einsum('qd,cd->qc', queries, centroids)

print(f'atten: {atten.shape}')