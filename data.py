import torch 

def load_text(path: str): 
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
    
def build_vocab(text: str):
    all_chars = sorted(set(text))
    stoi = {}
    itos = {}
    for (i, char) in enumerate(all_chars):
        stoi[char] = i
        itos[i] = char

    return stoi, itos

def encode(text: str, stoi: dict[str, int]):
    return torch.tensor([stoi[c] for c in text], dtype=torch.int64)

def decode(ids, itos: dict[int, str]):
    return "".join([itos[i] for i in ids.tolist()])

def train_val_split(ids: torch.Tensor, val_frac: float = 0.1): 
    split_point = int((1-val_frac) * ids.shape[0])
    return (ids[:split_point], ids[split_point:])

def get_batch(
        data: torch.Tensor, 
        batch_size: int, 
        block_size: int, 
        device: torch.device | str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    start_ids = torch.randint(0, data.shape[0] - block_size, (batch_size,))

    xs = torch.empty(batch_size, block_size, device=data.device, dtype=torch.int64)
    ys = torch.empty(batch_size, block_size, device=data.device, dtype=torch.int64)

    for i, start_id in enumerate(start_ids.tolist()): 
        xs[i] = data[start_id: start_id + block_size]
        ys[i] = data[start_id + 1: start_id + block_size + 1]

    if device is None: 
        device = data.device

    return xs.to(device=device, non_blocking=True), ys.to(device=device, non_blocking=True)