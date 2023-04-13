import torch

x = torch.rand(size=(4, 3, 2))
print(x.shape)
x = x.unsqueeze(dim=1)
print(x.shape)

c = torch.randn(size=(1024,)).tile(10, 1, 1)
print('c.shape', c.shape)

T = torch.tensor([1.0, 2.0, 3.0, 4.0])
T = T[None, None, :].repeat(10, 1, 1)
print('T.shape', T.shape)

linear = torch.nn.Linear(772, 768)
# print(linear(T))
