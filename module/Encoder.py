from torch import nn

from module.EncoderLayer import TransformerEncoderLayer


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_size, num_heads, ff_hidden_size, dropout, device):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_size, num_heads, ff_hidden_size, dropout, device)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        """
        :param mask: [batch_size, seq_len, seq_len]
        :param x: [batch_size, seq_len, embed_size]
        :return: [batch_size, seq_len, embed_size]
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x
