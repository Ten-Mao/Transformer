from torch import nn


class FeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size, device):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size).to(device)
        self.relu = nn.ReLU().to(device)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size).to(device)

    def forward(self, x):
        """
        :param x: [batch_size, seq_len, embed_size]
        :return: [batch_size, seq_len, embed_size]
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
