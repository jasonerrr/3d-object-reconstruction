import torch

a_cpu = torch.tensor([1.0, 2.0, 3.0])
print('device of a_cpu:', a_cpu.device)
a_cuda = a_cpu.to('cuda:0')
print('device of a_cpu:', a_cpu.device)
print('device of a_cuda:', a_cuda.device)