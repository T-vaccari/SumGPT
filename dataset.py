from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from adapter import RECORD_LENGTH, encode, format_record


def build_records(seed: int) -> np.ndarray:
    pairs = np.arange(1_000_000, dtype=np.int32)
    np.random.default_rng(seed).shuffle(pairs)
    records = np.empty((len(pairs), RECORD_LENGTH), dtype=np.uint8)
    for row, pair in enumerate(pairs):
        records[row] = encode(format_record(int(pair // 1000), int(pair % 1000)))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    args = parser.parse_args()

    records = build_records(args.seed)
    split = int(len(records) * (1 - args.validation_fraction))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "train.npy", records[:split])
    np.save(args.output_dir / "val.npy", records[split:])
    print(f"wrote {split} training and {len(records) - split} validation records to {args.output_dir}")


if __name__ == "__main__":
    main()
