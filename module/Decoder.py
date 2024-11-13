from torch import nn

from module.DecoderLayer import TransformerDecoderLayer


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_size, num_heads, ff_hidden_size, dropout, device):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_size, num_heads, ff_hidden_size, dropout, device)
            for _ in range(num_layers)
        ])

    def forward(self, x, enc_out, src_mask, tgt_mask):
        """
        :param x: [batch_size, tgt_len, embed_size]
        :param enc_out: [batch_size, src_len, embed_size]
        :param src_mask: [batch_size, src_len, src_len]
        :param tgt_mask: [batch_size, tgt_len, tgt_len]
        :return: [batch_size, tgt_len, embed_size]
        """
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x