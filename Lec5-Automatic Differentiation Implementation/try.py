import torch

x = torch.rand(2, 3, 4)
shape = x.shape
shape_out = [1] + [5]
print(shape_out)