import torch
from sympy import false
from torch import nn
from transformers.utils.fx import torch_unsqueeze


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, device, padding_idx=None):
        super(TokenEmbedding, self).__init__()
        if padding_idx is not None:
            self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=padding_idx).to(device)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size).to(device)
        self.embed_size = embed_size

    def forward(self, tokens_ids):
        """

        :param tokens_ids: [batch_size, seq_len]
        :return:      [batch_size, seq_len, embed_size]
        """
        return self.embedding(tokens_ids)

class PositionEmbedding(nn.Module):
    def __init__(self, max_len, embed_size, device):
        super(PositionEmbedding, self).__init__()
        self.embedding = torch.zeros(max_len, embed_size).to(device)
        self.embedding.requires_grad = False
        position = torch.arange(0, max_len).unsqueeze(dim=-1).to(device)
        _2i = torch.arange(0, embed_size, 2).unsqueeze(dim=0).to(device)
        self.embedding[:, 0::2] = torch.sin(position / 10000 ** (_2i / embed_size))
        self.embedding[:, 1::2] = torch.cos(position / 10000 ** (_2i / embed_size))
    def forward(self, tokens_ids):
        """

        :param tokens_ids: [batch_size, seq_len]
        :return:      [batch_size, seq_len, embed_size]
        """
        return self.embedding[:tokens_ids.size(1), :].unsqueeze(0).expand(tokens_ids.size(0), -1, -1)

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, device, max_len=1024, padding_idx=None):
        super(TransformerEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(vocab_size, embed_size, device, padding_idx)
        self.position_embedding = PositionEmbedding(max_len, embed_size, device)
        self.embed_size = embed_size

    def forward(self, tokens_ids):
        """

        :param tokens_ids: [batch_size, seq_len]
        :return:      [batch_size, seq_len, embed_size]
        """
        return self.token_embedding(tokens_ids) + self.position_embedding(tokens_ids)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    embed = TransformerEmbedding(1000, 512, 100, device)
    tokens = torch.randint(0, 1000, [32, 10]).to(device)
    out = embed(tokens)