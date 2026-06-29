import random
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N       = 1000000
REVERSE = True
SEED    = 42
# ---------------------------------------------------------------------------

def generate(n=N, reverse=REVERSE, seed=SEED):
    random.seed(seed)
    pairs = set()
    while len(pairs) < n:
        a = random.randint(0, 999)
        b = random.randint(0, 999)
        pairs.add((a, b))

    lines = []
    for a, b in pairs:
        result = str(a + b).zfill(4)
        if reverse:
            result = result[::-1]
        lines.append(f"{str(a).zfill(3)}+{str(b).zfill(3)}={result}")

    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    text = generate()

    # vocab
    chars  = sorted(list(set(text)))
    stoi   = {ch: i for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]

    # encode and split
    ids = np.array(encode(text), dtype=np.uint16)
    n   = int(0.9 * len(ids))

    np.array(ids[:n]).tofile('train.bin')
    np.array(ids[n:]).tofile('val.bin')

    print(f"Total tokens : {len(ids):,}")
    print(f"Train tokens : {n:,}")
    print(f"Val tokens   : {len(ids) - n:,}")
    print(f"Vocab        : {chars}")
    print(f"Example      : {text.splitlines()[0]}")