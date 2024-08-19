import transformer
import torch

query = torch.randn(64, 10, 512)
key = torch.randn(64, 10, 512)
value = torch.randn(64, 10, 512)

attention = transformer.MultiHeadAttention(512, 8, 0.1)
output = attention(query, key, value)
print(output.shape)
