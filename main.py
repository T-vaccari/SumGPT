import torch
from model import SumGPT

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
device   = 'cuda'
ckpt     = 'checkpoints/ckpt_best.pt'
reverse  = True    # must match how data.py was run
# ---------------------------------------------------------------------------

# vocab — hardcoded, same as training
chars  = sorted(list("\n0123456789+="))
stoi   = {ch: i for i, ch in enumerate(chars)}
itos   = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# ---------------------------------------------------------------------------
# Load model from checkpoint
# ---------------------------------------------------------------------------
state    = torch.load(ckpt, map_location=device, weights_only=True)
config   = state['config']

model = SumGPT(
    vocab_size = config['vocab_size'],
    n_embd     = config['n_embd'],
    n_head     = config['n_head'],
    block_size = config['block_size'],
    n_blocks   = config['n_blocks'],
    dropout    = config['dropout'],
).to(device)

model.load_state_dict(state['model'])
model.eval()
print(f"Loaded checkpoint from step {state['step']} | val loss {state['best_val_loss']:.4f}\n")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def predict(a: int, b: int) -> str:
    prompt  = f"{str(a).zfill(3)}+{str(b).zfill(3)}="
    idx     = torch.tensor([encode(prompt)], dtype=torch.long, device=device)  # (1, T)
    output  = model.generate(idx, max_new_tokens=5, decode=decode)
    # extract only the generated part after '='
    generated = decode(output[0].tolist())[len(prompt):]
    generated = generated.split('\n')[0]   # stop at newline

    result_digits = generated[::-1] if reverse else generated
    predicted = int(result_digits) if result_digits.isdigit() else None
    correct   = a + b

    status = '✓' if predicted == correct else '✗'
    return f"{prompt}{generated}  →  {status}  (expected {correct})"


# ---------------------------------------------------------------------------
# Run some tests
# ---------------------------------------------------------------------------
tests = [
    (1,   2),
    (123, 456),
    (999, 999),
    (0,   0),
    (100, 900),
]

print("--- spot checks ---")
for a, b in tests:
    print(predict(a, b))

print("--- interactive (q to quit) ---")
while True:
    try:
        raw = input("\na b: ").strip()
        if raw == 'q':
            break
        a, b = map(int, raw.split())
        if not (0 <= a <= 999 and 0 <= b <= 999):
            print("numbers must be in range 0-999")
            continue
        print(predict(a, b))
    except ValueError:
        print("enter two integers separated by space, e.g. 123 456")