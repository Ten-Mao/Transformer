from torch import nn

from module.FeedForward import FeedForward
from module.LayerNorm import LayerNorm
from module.MultiHeadAttention import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout, device):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(embed_size, num_heads, device)
        self.dropout1 = nn.Dropout(dropout).to(device)
        self.norm1 = LayerNorm(embed_size, device)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size, device)
        self.dropout2 = nn.Dropout(dropout).to(device)
        self.norm2 = LayerNorm(embed_size, device)

    def forward(self, x, mask=None):
        """
        :param mask: [batch_size, seq_len, seq_len]
        :param x: [batch_size, seq_len, embed_size]
        :return: [batch_size, seq_len, embed_size]
        """
        attention = self.self_attention(x, x, x, mask)
        x = x + self.dropout1(attention)
        x = self.norm1(x)

        feed_forward = self.feed_forward(x)
        x = x + self.dropout2(feed_forward)
        x = self.norm2(x)
        return x