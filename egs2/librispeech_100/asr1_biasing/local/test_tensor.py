import torch

a = torch.tensor([[2, 2, 0, 2, 3, 4, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 4, 4, 4, 4, 4,
         4, 3, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
         0, 0, 3, 3, 3, 4, 4, 0, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2,
         2, 3, 3, 3, 4, 3, 4, 4, 1, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3,
         3, 3, 3, 1, 3, 3, 3, 1, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 3, 3, 1, 1, 1, 1,
         1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 2, 4, 4, 4, 2, 2, 2, 2, 2, 1, 0,
         1, 2, 2, 1, 1, 1, 3, 0, 1, 3, 1, 1, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
         2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
         0, 3, 3, 1, 3, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 3, 3, 1, 3, 1, 1, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 1, 2, 3, 3, 3, 3, 3, 1,
         1, 1, 3, 2, 2, 2, 2, 3, 3, 3, 3, 2, 3, 3, 1, 1, 1, 3, 1, 4, 3, 3, 3, 4,
         4, 4, 4, 4, 2, 4, 4, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 1, 3, 3,
         3, 3, 4, 4, 4, 4, 4, 3, 4, 4, 0, 0],
        [2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 2, 3, 2, 2, 2, 2, 3, 3, 3, 1, 1, 0, 3,
         2, 0, 0, 0, 0, 2, 0, 0, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
         1, 4, 4, 1, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2,
         2, 2, 2, 0, 0, 0, 0, 2, 0, 0, 0, 3, 3, 3, 3, 3, 2, 2, 2, 3, 2, 3, 2, 0,
         0, 2, 2, 0, 2, 3, 3, 3, 3, 0, 3, 3, 3, 3, 4, 3, 3, 2, 2, 1, 2, 2, 2, 2,
         2, 2, 2, 2, 2, 2, 2, 0, 2, 1, 1, 0, 0, 0, 4, 4, 4, 4, 0, 0, 0, 3, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 4, 4, 0, 0, 0, 0, 3, 1,
         3, 3, 3, 3, 3, 3, 4, 4, 0, 0, 1, 1, 3, 2, 2, 2, 2, 3, 3, 3, 0, 4, 3, 3,
         3, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 1,
         2, 1, 1, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
         1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 0, 2, 0, 0, 0, 0, 1, 3, 3, 1, 1, 1, 2, 1,
         2, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2, 2,
         3, 2, 2, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 4, 0, 0, 2, 3, 3, 3, 2, 2, 3, 3,
         3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 2, 2, 2, 2, 3, 2, 3, 3, 0,
         1, 1, 3, 1, 1, 1, 4, 4, 4, 4, 0, 4]])

clusters = 5

b = torch.arange(0, clusters * 2, clusters).view(-1, 1)
c = a + b
print(c)

c, count = c.unique(sorted=True, return_counts=True)

print(c.view(2, -1) - b)
print(count.view(2, -1))
# print(a[0, :].unique())
# print(a[1, :].unique())