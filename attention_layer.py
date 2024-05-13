import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionLayer(nn.Module):
    def __init__(self):
        super(SelfAttentionLayer, self).__init__(1)
        self.embed_size = 1536
        self.heads = 12
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = self.values(values).view(N, value_len, self.heads, self.embed_size // self.heads)
        keys = self.keys(keys).view(N, key_len, self.heads, self.embed_size // self.heads)
        queries = self.queries(query).view(N, query_len, self.heads, self.embed_size // self.heads)

        values = values.transpose(1, 2)  # (N, heads, value_len, head_dim)
        keys = keys.transpose(1, 2)  # (N, heads, key_len, head_dim)
        queries = queries.transpose(1, 2)  # (N, heads, query_len, head_dim)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).transpose(1, 2)
        out = out.contiguous().view(N, query_len, self.embed_size)

        out = self.fc_out(out)
        return out


# Parameters
embed_size = 128 * 12  # Total embedding size 1536
heads = 12

# Example initialization and forward pass
self_attention = SelfAttentionLayer()
x = torch.rand(64, 10, embed_size)  # Example input (batch_size, sequence_length, embed_size)
mask = None  # Example mask, replace with actual mask as needed
output = self_attention(x, x, x, mask)
print(len(output))
