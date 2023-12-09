import torch

B, N, D = 5, 3, 4
V = 10

embeds = torch.randn(B, N, D)

print(embeds)
print(embeds.shape)

embeds = embeds.reshape(B * N, D)
print(embeds.shape)

index = torch.randint(0, V, (B, N))
print(index)
print(index.shape)
index = index.reshape(B * N)
print(index.shape)

x = torch.zeros(V, D)
x.index_add_(0, index, embeds)
print(x)

count = torch.zeros(V)
count.index_add_(0, index, torch.ones(B * N))
print(count)

x = x / count.unsqueeze(1)
print(x)