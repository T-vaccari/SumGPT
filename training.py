import os
import numpy as np
import torch
from model import SumGPT

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
batch_size    = 512
block_size    = 12
example_len   = 13      # block_size + \n
max_iters     = 10000
eval_interval = 250
eval_iters    = 100
learning_rate = 3e-4
n_embd        = 128
n_head        = 4
n_blocks      = 4
dropout       = 0.0
device        = 'cuda'
ckpt_dir      = 'checkpoints'
RESUME = False   # set True to continue from ckpt_last.pt
# ---------------------------------------------------------------------------

torch.manual_seed(1337)
os.makedirs(ckpt_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Data — memory mapped, never fully loaded into RAM
# ---------------------------------------------------------------------------
train_data = np.memmap('train.bin', dtype=np.uint16, mode='r')
val_data   = np.memmap('val.bin',   dtype=np.uint16, mode='r')

# vocab — small enough to hardcode
chars      = sorted(list("\n0123456789+="))
vocab_size = len(chars)
stoi       = {ch: i for i, ch in enumerate(chars)}
itos       = {i: ch for i, ch in enumerate(chars)}
encode     = lambda s: [stoi[c] for c in s]
decode     = lambda l: ''.join([itos[i] for i in l])
eq_id      = stoi['=']   


def get_batch(split):
    data       = train_data if split == 'train' else val_data
    n_examples = (len(data) - block_size) // example_len
    ix         = torch.randint(n_examples, (batch_size,)) * example_len
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64))         for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64)) for i in ix])

    # ignora nella loss tutto ciò che precede '=': operandi casuali e '+'
    # non sono apprendibili — solo le cifre della somma (e il \n finale) contano
    eq_pos  = (x == eq_id).float().argmax(dim=1)
    col_idx = torch.arange(block_size).unsqueeze(0)
    mask    = col_idx < eq_pos.unsqueeze(1)
    y[mask] = -100

    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Model & optimizer
# ---------------------------------------------------------------------------
model     = SumGPT(vocab_size, n_embd, n_head, block_size, n_blocks, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------
@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y      = get_batch(split)
            _, loss   = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------
def save_checkpoint(path, step, best_val_loss):
    tmp = path + '.tmp'
    torch.save({
        'model':         model.state_dict(),
        'optimizer':     optimizer.state_dict(),
        'step':          step,
        'best_val_loss': best_val_loss,
        'config': {
            'vocab_size': vocab_size,
            'n_embd':     n_embd,
            'n_head':     n_head,
            'block_size': block_size,
            'n_blocks':   n_blocks,
            'dropout':    dropout,
        },
    }, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
best_val_loss = float('inf')

if RESUME:
    ckpt          = torch.load(f'{ckpt_dir}/ckpt_last.pt', map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    start_step    = ckpt['step']
    best_val_loss = ckpt['best_val_loss']
    print(f"Resuming from step {start_step} | best val {best_val_loss:.4f}")

for step in range(max_iters):

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step:4d} | train {losses['train']:.4f} | val {losses['val']:.4f}")

        save_checkpoint(f'{ckpt_dir}/ckpt_last.pt', step, best_val_loss)
        if losses['val'] < best_val_loss:
            best_val_loss = losses['val']
            save_checkpoint(f'{ckpt_dir}/ckpt_best.pt', step, best_val_loss)
            print(f"           → best val {best_val_loss:.4f} saved")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Done.")