import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, device):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_dim = embed_size // num_heads
        assert self.head_dim * num_heads == embed_size, "Embed size needs to be divisible by num_heads"

        for i in range(num_heads):
            setattr(self, f'W_q_{i}', nn.Linear(self.embed_size, self.head_dim).to(device))
            setattr(self, f'W_k_{i}', nn.Linear(self.embed_size, self.head_dim).to(device))
            setattr(self, f'W_v_{i}', nn.Linear(self.embed_size, self.head_dim).to(device))
        self.W_o = nn.Linear(num_heads * self.head_dim, embed_size).to(device)

    def forward(self, q, k, v, mask):
        """
        :param q: [batch_size, query_len, embed_size]
        :param k: [batch_size, key_len, embed_size]
        :param v: [batch_size, value_len, embed_size]
        :param mask: [batch_size, query_len, key_len]
        :return: [batch_size, query_len, embed_size]
        """
        query_len = q.shape[1]
        key_len = k.shape[1]
        value_len = v.shape[1]

        assert key_len == value_len, "Key len and value len need to be equal"

        # Split the embedding into self.num_heads different pieces
        # And then combine them together at the end
        heads = []
        for i in range(self.num_heads):
            W_q = getattr(self, f'W_q_{i}')
            W_k = getattr(self, f'W_k_{i}')
            W_v = getattr(self, f'W_v_{i}')

            query_i = W_q(q)
            key_i = W_k(k)
            value_i = W_v(v)

            # Scaled dot product attention
            scaled_attention_logits = query_i @ key_i.transpose(-2, -1) / (self.head_dim ** 0.5)
            if mask is not None:
                scaled_attention_logits += (mask * -1e9)

            attention_weights = scaled_attention_logits.softmax(dim=-1)
            attention = attention_weights @ value_i
            heads.append(attention)

        # Combine all the attention heads together
        heads = torch.cat(heads, dim=-1)
        heads = self.W_o(heads)
        return heads


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    multi_head_attn = MultiHeadAttention(512, 8, device)
    q = torch.rand(32, 10, 512).to(device)
    k = torch.rand(32, 10, 512).to(device)
    v = torch.rand(32, 10, 512).to(device)
    mask = torch.randint(0, 2, [32, 10, 10]).to(device).bool()
    out = multi_head_attn(q, k, v, mask)
    print(out.shape)