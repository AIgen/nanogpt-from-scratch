import torch
import math
from typing import Callable
from nanogpt import GPT, GPTConfig, compute_loss
from data import get_batch, load_text, build_vocab, encode, train_val_split

def _train_core(
        model: GPT, 
        num_steps: int, 
        next_batch: Callable,
        evaluator: Callable | None = None,
        eval_every: int = 50,
        lr: float = 3e-4, 
        lr_fn: Callable | None = None,
        weight_decay: float = 0.0, 
        grad_clip: float | None = 1.0,
        device: torch.device | str = "cpu"
):
    model.train()
    model.to(device=device)
    params = list(model.parameters())

    optim = torch.optim.AdamW(
        params = params,
        lr=lr, 
        weight_decay=weight_decay,   
    )

    losses = []
    if evaluator:
        eval_losses = []
    for step in range(num_steps): 
        if evaluator is not None and (step % eval_every) == 0:
            eval_losses.append(evaluator())

        optim.zero_grad() # set gradient info to 0 for each parameter tracked by optim
        if lr_fn is not None: 
            optim.param_groups[0]['lr'] = lr_fn(step)
        xs, ys = next_batch()
        loss = compute_loss(model, xs, ys) # get loss on batch
        loss.backward() # populate gradients based on loss 
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(
                params, 
                max_norm=grad_clip,
            )
        optim.step() # take step based on gradients, hyperamaters, and algorithm
        losses.append(loss.item())
        
    if evaluator: 
        return losses, eval_losses
    else: 
        return losses

def train(
        model: GPT, 
        data: torch.Tensor, 
        num_steps: int, 
        batch_size: int,
        block_size: int,
        evaluator: Callable | None = None,
        eval_every: int = 50,
        lr: float = 3e-4, 
        lr_fn: Callable | None = None,
        weight_decay: float = 0.0, 
        grad_clip: float | None = 1.0,
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
        evaluator=evaluator,
        eval_every=eval_every,
        lr = lr, 
        lr_fn = lr_fn,
        weight_decay = weight_decay,
        grad_clip=grad_clip,
        device = device,
    )


def overfit_one_batch(
        model: GPT, 
        data: torch.Tensor, 
        num_steps: int, 
        batch_size: int,
        block_size: int,
        lr: float = 3e-4, 
        evaluator: Callable | None = None,        
        eval_every: int = 50,
        weight_decay: float = 0.0, 
        grad_clip: float | None = 1.0,
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
        evaluator=evaluator,
        eval_every=eval_every,
        lr = lr, 
        weight_decay = weight_decay,
        grad_clip=grad_clip,
        device = device,
    )


def evaluate(
        model: GPT, 
        data: torch.Tensor, 
        batch_size: int, 
        block_size: int, 
        num_batches: int = 10, 
) -> float: 
    model.eval()

    tot_loss = 0.0
    with torch.no_grad():
        for _ in range(num_batches): 
            xs, ys = get_batch(
                data, 
                batch_size=batch_size,
                block_size=block_size,
            )

            loss = compute_loss(model, xs, ys).item()
            tot_loss += loss

    model.train()
    return tot_loss/num_batches


def cosine_lr(
        step: int, 
        total_steps: int, 
        warmup_steps: int,
        lr_min: float, 
        lr_max: float, 
) -> float: 
    if step < warmup_steps: 
        return lr_min + (step/warmup_steps) * (lr_max-lr_min)
    if step >= total_steps: 
        return lr_min
    
    step_shift = step - warmup_steps
    tot_step_shift = total_steps - warmup_steps
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * step_shift/tot_step_shift))

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
    batch_size = 32
    model = GPT(config)
    evaluator = lambda: evaluate(
        model=model,
        data=val_data,
        batch_size=batch_size,
        block_size=config.block_size,
        num_batches=10,
    )
    
    losses_train, losses_val = train(
        model=model,
        data=train_data, 
        num_steps=200, 
        batch_size=batch_size, 
        block_size=config.block_size,
        evaluator=evaluator,
    )

    print("Train losses: ", losses_train[::20])
    print("Val losses: ", losses_val)

