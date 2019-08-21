import torch
import torch.nn.functional as F
model = torch.nn.Linear(512,1)

a = torch.rand(1,1,10)
b = F.softmax(a, dim=2)
print(b.size())