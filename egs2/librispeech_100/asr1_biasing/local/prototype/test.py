import torch


samples = torch.Tensor([
    [0.1, 0.1],    #-> group / class 1
    [0.2, 0.2],    #-> group / class 2
    [0.4, 0.4],    #-> group / class 2
    [0.0, 0.0]     #-> group / class 0
])

labels = torch.LongTensor([1, 2, 2, 0])
labels = labels.view(1, labels.size(0), 1).expand(-1, -1, samples.size(1))
print(labels)
print(f'labels shape: {labels.shape}')

unique_labels, labels_count = labels.unique(dim=0, return_counts=True)

res = torch.zeros_like(unique_labels, dtype=torch.float).scatter_add_(1, labels, samples)
res = res / labels_count.float().unsqueeze(1)