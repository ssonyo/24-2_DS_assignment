import torch
import torch.nn as nn
import math
from torch import Tensor, float

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.embedding(x)

class PositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super(PositionEmbedding, self).__init__()

        pe = torch.zeros(max_len, d_model)  #TODO
        position = torch.arange(0, max_len, dtype=float).unsqueeze(1) # shape: (max_len,1)
        div_term = 1 / (10000.0 ** (torch.arange(0, d_model, 2)).float() / d_model)  # 10000^(2i/d_model)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # register_buffer을 이용해서 학습은 안되지만, model의 state에서 움직이도록 할 수 있다.
        self.register_buffer('position_encoding', pe.unsqueeze(0))  # shape; (1, max_len, d_model). 
        # batch 차원을 만들어줘야 나중에 tokenembedding 결과와 더할때 broadcasting 가능


    def forward(self, x: Tensor) -> Tensor:
        return self.position_encoding[:, :x.shape[1], :]  # TODO one line!