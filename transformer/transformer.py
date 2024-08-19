import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    # hid_dim: the dimension of the input
    # n_head: the number of heads
    def __init__(self, hid_dim, n_head, dropout):
        super(MultiHeadAttention, self).__init__()

        self.hid_dim = hid_dim
        self.n_head = n_head
        # force the hid_dim to be divisible by n_head
        assert hid_dim % n_head == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_head]))

    # In transformer, the query, key, and value are the same input
    # query: [batch size, query len, hid dim]
    # key: [batch size, key len, hid dim]
    # value: [batch size, value len, hid dim]
    # mask: [batch size, n head, query len, key len]
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        # split the hid_dim into n_head
        Q = Q.view(batch_size, -1, self.n_head, self.hid_dim // self.n_head).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_head, self.hid_dim // self.n_head).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_head, self.hid_dim // self.n_head).permute(0, 2, 1, 3)

        attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e10)
        attention = F.softmax(attention, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.hid_dim)
        x = self.fc(x)

        return x
