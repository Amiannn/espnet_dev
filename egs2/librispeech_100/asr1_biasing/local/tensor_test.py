import os
import torch
import numpy as np

from tqdm  import tqdm

# from local.utils import read_file
# from local.utils import read_json
# from local.utils import write_json
# from local.utils import write_file

if __name__ == "__main__":
    B, L, D = 2, 3, 2

    x = torch.randn(B, L, D)
    print(x)
    print(x.shape)

    indices = torch.randint(0, L, (B, 1, 1)).repeat(1, 1, D)
    # indices = torch.tensor([[[0, 0]], [[1, 1]]])
    print(indices)
    print(indices.shape)
    
    y = torch.gather(x, 1, indices)
    print(y)
    print(y.shape)


    indices = torch.tensor([
        6, 3, 2, 5, 6, 3, 6, 5, 5, 2, 6, 3, 3, 4, 5, 5, 3, 4, 4, 4, 4, 5, 5, 4,
        4, 4, 5, 3, 2, 4, 2, 7, 5, 6, 5, 3, 2, 6, 3, 4, 4, 4, 5, 5, 2, 4, 4, 5,
        4, 3, 5, 4, 6, 3, 4, 2, 5, 4, 5, 4, 4, 5, 5, 3, 4, 4, 3, 5, 5, 5, 4, 3,
        3, 7, 4, 3, 4, 3, 3, 2, 3, 4, 6, 2, 3, 2, 3, 7, 4, 7, 6, 3, 5, 2, 6, 7,
        5, 4, 3, 4, 6, 4, 3, 4, 3, 6, 5, 5, 3, 5, 3, 4, 4, 3, 6, 7, 5, 2, 5, 5,
        4, 6, 2, 7, 3, 6, 7, 4, 6, 7, 4, 5, 6, 3, 5, 5, 4, 3, 8, 3, 4, 3, 5, 5,
        4, 6, 4, 4, 5, 5, 6, 7, 6, 5, 4, 4, 5, 4, 3, 7, 3, 4, 4, 3, 7, 4, 3, 7,
        3, 5, 4, 7, 4, 2, 5, 3, 4, 3, 2, 5, 4, 6, 5, 5, 5, 4, 4, 7, 6, 5, 3, 4,
        4, 5, 5, 4, 7, 3, 7, 6
    ])
    print(torch.max(indices))
    x = torch.randn(200, 8, 256)
    y = x[torch.arange(200), indices]
    print(y)
