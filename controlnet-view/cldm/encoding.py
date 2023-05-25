import torch
import torch.nn as nn
import torch.nn.functional as F

class FreqEncoder_torch(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):
    
        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2 ** torch.linspace(0, max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2 ** 0, 2 ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, **kwargs):

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                # print('input =', input, type(input))
                # print('freq =', freq, type(freq))
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)

        return out


if __name__ == '__main__':
    xxx = torch.tensor([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    xxxe = FreqEncoder_torch(3, 6, 7)(xxx)
    print(xxxe)
    print(xxxe.shape)
    print(xxxe.view(2, -1, 3).permute(0, 2, 1), xxxe.view(2, -1, 3).permute(0, 2, 1).shape)

