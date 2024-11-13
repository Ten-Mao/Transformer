from torch import nn

from module.FeedForward import FeedForward
from module.MultiHeadAttention import MultiHeadAttention


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout, device):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention1 = MultiHeadAttention(embed_size, num_heads, device)
        self.dropout1 = nn.Dropout(dropout).to(device)
        self.norm1 = nn.LayerNorm(embed_size).to(device)
        self.self_attention2 = MultiHeadAttention(embed_size, num_heads, device)
        self.dropout2 = nn.Dropout(dropout).to(device)
        self.norm2 = nn.LayerNorm(embed_size).to(device)
        self.feed_forward = FeedForward(embed_size, ff_hidden_size, device)
        self.dropout3 = nn.Dropout(dropout).to(device)
        self.norm3 = nn.LayerNorm(embed_size).to(device)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        :param x: [batch_size, tgt_len, embed_size]
        :param encoder_output: [batch_size, src_len, embed_size]
        :param src_mask: [batch_size, src_len, src_len]
        :param tgt_mask: [batch_size, tgt_len, tgt_len]
        :return: [batch_size, tgt_len, embed_size]
        """
        attention = self.self_attention1(x, x, x, tgt_mask)
        x = x + self.dropout1(attention)
        x = self.norm1(x)

        attention = self.self_attention2(x, encoder_output, encoder_output, src_mask)
        x = x + self.dropout2(attention)
        x = self.norm2(x)

        feed_forward = self.feed_forward(x)
        x = x + self.dropout3(feed_forward)
        x = self.norm3(x)
        return x
