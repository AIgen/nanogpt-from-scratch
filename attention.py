import torch
from torch import nn
from nanogpt import GPTConfig


def scaled_dot_product_attention(
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor, 
        mask: torch.Tensor | None = None,
) -> torch.Tensor:
    T = q.shape[-2]
    sqrt_d_k = q.shape[-1]**0.5
    qkT = (q / sqrt_d_k ) @ k.mT
    if mask is not None: 
        qkT = qkT + mask[:T, :T]
    res = qkT.softmax(dim=-1) @ v
    
    return res

class SingleHeadCausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig): 
        super().__init__()
        n_embd = config.n_embd
        d_k = config.n_embd

        self.W_q = nn.Linear(n_embd, d_k, bias=False)
        self.W_k = nn.Linear(n_embd, d_k, bias=False)
        self.W_v = nn.Linear(n_embd, n_embd, bias=False)
        self.W_o = nn.Linear(n_embd, n_embd, bias=False)

        mask = - torch.triu(1.0 / torch.zeros(config.block_size, config.block_size), diagonal = 1)
        self.register_buffer("mask", mask, persistent=False)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        # x is (B, T, n_embd), so we expect that W.shape[-2] is n_embed
        # Then, x @ W is (B, T, W.shape[-1])
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        res = self.W_o(scaled_dot_product_attention(q, k, v, self.mask))

        return res

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig): 
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.d_k = int(config.n_embd/config.n_head)
        self.block_size = config.block_size

        self.W_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.W_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.W_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.W_o = nn.Linear(self.n_embd, self.n_embd, bias=False)

        mask = torch.triu(-1.0 / torch.zeros(self.block_size, self.block_size), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor): 
        shape_ = list(x.shape)
        shape_split = shape_[:-1] + [self.n_head, self.d_k]
        q = self.W_q(x).reshape(shape_split).transpose(-3, -2)
        k = self.W_k(x).reshape(shape_split).transpose(-3, -2)
        v = self.W_v(x).reshape(shape_split).transpose(-3, -2)
        
        multihead = scaled_dot_product_attention(q, k, v, self.mask).transpose(-3, -2).reshape(shape_)
        res = self.W_o(multihead)
        return res
