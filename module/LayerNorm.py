import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, embed_size, device, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gama = nn.Parameter(torch.ones(embed_size)).to(device)
        self.beta = nn.Parameter(torch.zeros(embed_size)).to(device)
        self.eps = eps
    def forward(self, x):
        """
        :param x: [batch_size, seq_len, embed_size]
        :return: [batch_size, seq_len, embed_size]
        """
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        return self.gama * (x - mean) / torch.sqrt(var + self.eps) + self.beta
