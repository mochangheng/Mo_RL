import torch.nn as nn

net = nn.Sequential(
    nn.Linear(4, 8),
    nn.Linear(8, 16),
)

params = list(net.parameters())[0]
print(params+1)