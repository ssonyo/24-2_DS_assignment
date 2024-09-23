import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional, Tuple

class QueryLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(QueryLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class KeyLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(KeyLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ValueLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(ValueLayer, self).__init__()
        self.linear = nn.Linear(d_model, d_model * n_heads)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class ScaledDotProductAttention(nn.Module):
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        #TODO
        # q shape; (batch_size, n_heads, sequence_length, d_k)

        d_k = q.shape[-1]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # shape; (batch, heads, seq_length, seq_length)

        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))

        attention_weight = F.softmax(scores, dim=-1)
        output = torch.matmul(attention_weight, v)

        return attention_weight, output


        
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        
        self.query_layers = QueryLayer(d_model, n_heads)
        self.key_layers = KeyLayer(d_model, n_heads)
        self.value_layers = ValueLayer(d_model, n_heads)
        self.attention = ScaledDotProductAttention()
        self.fc = nn.Linear(n_heads * d_model, d_model)
    
    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        #TODO
        batch_size = Q.size(0)
        d_k = self.d_model // self.n_heads

        # Linear projections for query, key, value
        Q = self.query_layers(Q)  # (batch_size, seq_len, d_model)
        K = self.key_layers(K)    # (batch_size, seq_len, d_model)
        V = self.value_layers(V)  # (batch_size, seq_len, d_model)

        # Reshape Q, K, V for multi-head attention
        # Split d_model into (n_heads, head_dim), where head_dim = d_model // n_heads
        Q = Q.view(batch_size, -1, self.n_heads, d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, d_k).transpose(1, 2)
        # After transpose: (batch_size, n_heads, seq_len, head_dim)

        # Apply scaled dot-product attention on each head
        attention_weights, attention_output = self.attention(Q, K, V, mask)
        
        # Concatenate the heads back together
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        print(4)
        print(attention_output.shape)
        print(self.d_model, self.n_heads)

        # Apply the final fully connected layer to combine heads' output
        output = self.fc(attention_output)
        print(5)
        
        return output