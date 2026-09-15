from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from adapter import CONTEXT_LENGTH, VOCABULARY, make_batch, masked_cross_entropy
from model import build_model


DEFAULT_CONFIG = {
    "vocab_size": len(VOCABULARY),
    "context_length": CONTEXT_LENGTH,
    "d_model": 128,
    "num_layers": 4,
    "num_heads": 4,
    "d_ff": 512,
    "rope_theta": 10000.0,
    "llm_library_commit": "ee3eb00b3826956ab518ccbd4a39abca172ad5f1",
}


@torch.inference_mode()
def estimate_loss(model, validation_records, batch_size, device, batches):
    model.eval()
    losses = []
    for _ in range(batches):
        inputs, targets = make_batch(validation_records, batch_size, device)
        losses.append(masked_cross_entropy(model(inputs), targets).item())
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/llm-library"))
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=250)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    train_records = torch.from_numpy(np.load(args.data_dir / "train.npy"))
    validation_records = torch.from_numpy(np.load(args.data_dir / "val.npy"))
    model = build_model(DEFAULT_CONFIG, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.95), weight_decay=0.1)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.checkpoint_dir / "config.json").write_text(json.dumps(DEFAULT_CONFIG, indent=2) + "\n")

    best_validation_loss = float("inf")
    for step in range(1, args.steps + 1):
        inputs, targets = make_batch(train_records, args.batch_size, device)
        optimizer.zero_grad(set_to_none=True)
        loss = masked_cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()
        if step % args.eval_interval == 0 or step == args.steps:
            validation_loss = estimate_loss(model, validation_records, args.batch_size, device, args.eval_batches)
            checkpoint = {"model": model.state_dict(), "config": DEFAULT_CONFIG, "step": step, "validation_loss": validation_loss}
            torch.save(checkpoint, args.checkpoint_dir / "last.pt")
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(checkpoint, args.checkpoint_dir / "best.pt")
            print(f"step={step} train_loss={loss.item():.4f} validation_loss={validation_loss:.4f}", flush=True)


if __name__ == "__main__":
    main()
