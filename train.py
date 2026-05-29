import torch
from typing import Callable
from nanogpt import GPT, GPTConfig, compute_loss
from data import get_batch, load_text, build_vocab, encode, train_val_split

def _train_core(
        model: GPT, 
        num_steps: int, 
        next_batch: Callable,
        lr: float = 3e-4, 
        weight_decay: float = 0.0, 
        device: torch.device | str = "cpu"
):
    model.train()
    model = model.to(device=device)
    params = model.parameters()
    optim = torch.optim.AdamW(
        params = params,
        lr=lr, 
        weight_decay=weight_decay,   
    )
    
    losses = []
    for _ in range(num_steps): 
        optim.zero_grad() # set gradient info to 0 for each parameter tracked by optim
        xs, ys = next_batch()
        loss = compute_loss(model, xs, ys) # get loss on batch
        loss.backward() # populate gradients based on loss 
        optim.step() # take step based on gradients, hyperamaters, and algorithm
        losses.append(loss.item())
        
    model.eval()
    return losses

def train(
        model: GPT, 
        data: torch.Tensor, 
        num_steps: int, 
        batch_size: int,
        block_size: int,
        lr: float = 3e-4, 
        weight_decay: float = 0.0, 
        device: torch.device | str = "cpu"
):
    def next_batch(): 
        return get_batch(
            data = data, 
            batch_size = batch_size, 
            block_size = block_size, 
            device = device,
        )
    return _train_core(
        model = model,
        num_steps = num_steps, 
        next_batch=next_batch,
        lr = lr, 
        weight_decay = weight_decay,
        device = device,
    )


def overfit_one_batch(
        model: GPT, 
        data: torch.Tensor, 
        num_steps: int, 
        batch_size: int,
        block_size: int,
        lr: float = 3e-4, 
        weight_decay: float = 0.0, 
        device: torch.device | str = "cpu"
):
    xs, ys = get_batch(
        data = data, 
        batch_size = batch_size, 
        block_size = block_size, 
        device = device,
    )  
    return _train_core(
        model = model,
        num_steps = num_steps, 
        next_batch=lambda: (xs, ys),
        lr = lr, 
        weight_decay = weight_decay,
        device = device,
    )

if __name__ == "__main__":
    data = load_text('input.txt')
    stoi, itos = build_vocab(data)
    encoded = encode(data, stoi)

    train_data, val_data = train_val_split(encoded)

    config = GPTConfig(
        vocab_size = len(stoi), 
        block_size = 64,
        n_layer = 4, 
        n_head = 4, 
        n_embd = 128
    )
    model = GPT(config)
    
    losses = train(
        model = model,
        data = encoded, 
        num_steps = 200, 
        batch_size = 32, 
        block_size = config.block_size,
    )

    print(losses[::20])
