from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from adapter import complete, decode
from model import build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/llm-library/best.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/val.npy"))
    parser.add_argument("--cases", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = build_model(checkpoint["config"], device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    generator = torch.Generator().manual_seed(args.seed)
    records = np.load(args.data)
    correct = 0
    for _ in range(args.cases):
        row = torch.randint(0, len(records), (1,), generator=generator).item()
        expected = decode(records[row].tolist())
        prediction = complete(model, expected[:8], device)
        correct += prediction == expected[:12]
    print(f"validation_accuracy={correct / args.cases:.2%} ({correct}/{args.cases})")


if __name__ == "__main__":
    main()
