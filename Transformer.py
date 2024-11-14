import torch
from torch import nn

from module.Decoder import TransformerDecoder
from module.Embedding import TransformerEmbedding
from module.Encoder import TransformerEncoder


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, embed_size, num_layers, num_heads, ff_hidden_size, dropout, device,
                 src_pad_idx=None, tgt_pad_idx=None, max_len=1024):
        super(Transformer, self).__init__()
        self.src_embedding = TransformerEmbedding(src_vocab_size, embed_size, device, max_len=max_len, padding_idx=src_pad_idx)
        self.tgt_embedding = TransformerEmbedding(tgt_vocab_size, embed_size, device, max_len=max_len, padding_idx=tgt_pad_idx)
        self.encoder = TransformerEncoder(num_layers, embed_size, num_heads, ff_hidden_size, dropout, device)
        self.decoder = TransformerDecoder(num_layers, embed_size, num_heads, ff_hidden_size, dropout, device)
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_vocab_size = tgt_vocab_size
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device
        self.embed_size = embed_size
        self.fc_out = nn.Linear(embed_size, tgt_vocab_size).to(device)

    def make_src_mask(self, src):
        """
        :param src: [batch_size, src_len]
        :return: [batch_size, src_len, src_len]
        """
        if self.src_pad_idx is None:
            mask = None
        else:
            mask = (src == self.src_pad_idx).unsqueeze(1).repeat(1, src.size(1), 1)
        return mask

    def make_tgt_mask(self, tgt):
        """
        :param tgt: [batch_size, tgt_len]
        :return: [batch_size, tgt_len, tgt_len]
        """
        tgt_len = tgt.size(1)
        tgt_future_mask = torch.triu(torch.ones(tgt_len, tgt_len, device=self.device), diagonal=1).repeat(tgt.size(0), 1, 1)
        if self.tgt_pad_idx is None:
            mask = tgt_future_mask
        else:
            tgt_pad_mask = (tgt == self.tgt_pad_idx).unsqueeze(1).repeat(1, tgt_len, 1)
            mask = tgt_future_mask + tgt_pad_mask
        return mask

    def make_cross_mask(self, src, tgt):
        """
        :param src: [batch_size, src_len]
        :param tgt: [batch_size, tgt_len]
        :return: [batch_size, tgt_len, src_len]
        """
        src_len = src.size(1)
        tgt_len = tgt.size(1)
        if self.src_pad_idx is None:
            cross_mask = None
        else:
            cross_mask = (src == self.src_pad_idx).unsqueeze(1).repeat(1, tgt_len, 1)
        return cross_mask

    def forward(self, src, tgt):
        """
        :param src: [batch_size, src_len]
        :param tgt: [batch_size, tgt_len]
        :return: [batch_size, tgt_len, tgt_vocab_size]
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        cross_mask = self.make_cross_mask(src, tgt)
        src_embed = self.src_embedding(src)
        tgt_embed = self.tgt_embedding(tgt)
        enc_out = self.encoder(src_embed, src_mask)
        dec_out = self.decoder(tgt_embed, enc_out, cross_mask, tgt_mask)
        out = self.fc_out(dec_out)
        return out