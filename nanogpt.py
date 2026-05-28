import torch 
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass 
from typing import Optional

@dataclass
class GPTConfig: 
        vocab_size: int
        block_size: int
        n_layer: int
        n_head: int
        n_embd: int
        dropout: float = 0.0

class GPTEmbeddings(nn.Module): 
    def __init__(
            self, 
            config: GPTConfig
    ):
        super().__init__()
        self.tok_emb = nn.Embedding(
            num_embeddings=config.vocab_size, 
            embedding_dim=config.n_embd
        )
        self.pos_emb = nn.Embedding(
            num_embeddings=config.block_size, 
            embedding_dim=config.n_embd
        )
        self.drop = nn.Dropout(config.dropout)

    def forward(self, idx: torch.Tensor):
        seq_len = idx.shape[1]
        assert seq_len <= self.pos_emb.num_embeddings
        tok = self.tok_emb(idx)
        positions = torch.arange(seq_len, device=idx.device)
        pos = self.pos_emb(positions)
        return self.drop(tok + pos)
    

class Block(nn.Module):
    def __init__(
            self, 
            config: GPTConfig
    ):
        super().__init__()
        n_embd = config.n_embd
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.Linear(n_embd, n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), 
            nn.GELU(), 
            nn.Linear(4*n_embd, n_embd),
        )

    def forward(
            self, 
            x: torch.Tensor,
    ):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        
        return x

        
class GPT(nn.Module):
    def __init__(
            self, 
            config: GPTConfig
    ):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.embeddings = GPTEmbeddings(config)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(self, idx: torch.Tensor):
        x = self.embeddings(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits

    @torch.no_grad()
    def generate(
            self,
            idx: torch.Tensor, 
            max_new_tokens: int,
            temperature: Optional[float] = None
    ) -> torch.Tensor:
        generation = torch.empty(
            idx.shape[0], 
            idx.shape[1] + max_new_tokens, 
            dtype=idx.dtype, 
            device=idx.device,
        )
        generation[:, :idx.shape[1]] = idx
        
        for gen_pos in range(idx.shape[1], idx.shape[1] + max_new_tokens):
            start_pos = max(0, gen_pos - self.block_size)
            if temperature is None or temperature == 0.0: 
                generation[:, gen_pos] = self.forward(
                    generation[:, start_pos: gen_pos]
                )[:, -1].argmax(dim=-1)
            else: 
                generation[:, gen_pos] = torch.multinomial(
                    (
                        self.forward(
                            generation[:, start_pos: gen_pos]
                        )[:, -1]/(temperature + 1e-6)
                    ).softmax(dim=1), 
                    num_samples=1,
                ).squeeze(-1)
        
        return generation


def compute_loss(
        model: GPT, 
        idx: torch.Tensor, 
        targets: torch.Tensor
) -> torch.Tensor:
    logits = model(idx) # (B, T, V)
    logits_flat = logits.reshape(-1, logits.shape[-1]) # (B * T, V)
    targets_flat = targets.reshape(-1) # (B * T)

    return F.cross_entropy(logits_flat, targets_flat)

if __name__ == "__main__":
    cfg = GPTConfig(vocab_size=50, block_size=8, n_layer=2, n_head=2, n_embd=16, dropout=0.0)
    model = GPT(cfg)
    idx = torch.randint(0, 50, (4, 8))
    logits = model(idx)
    assert logits.shape == (4, 8, 50)
