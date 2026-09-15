from __future__ import annotations

import torch
import torch.nn.functional as F


VOCABULARY = "\n0123456789+="
TOKEN_TO_ID = {token: index for index, token in enumerate(VOCABULARY)}
ID_TO_TOKEN = dict(enumerate(VOCABULARY))
IGNORE_INDEX = -100
RECORD_LENGTH = 13
CONTEXT_LENGTH = RECORD_LENGTH - 1


def encode(text: str) -> list[int]:
    return [TOKEN_TO_ID[token] for token in text]


def decode(token_ids: list[int]) -> str:
    return "".join(ID_TO_TOKEN[token_id] for token_id in token_ids)


def format_record(a: int, b: int) -> str:
    return f"{a:03d}+{b:03d}={(a + b):04d}\n"


def make_batch(records: torch.Tensor, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    indices = torch.randint(records.shape[0], (batch_size,))
    batch = records[indices].to(device=device, dtype=torch.long, non_blocking=True)
    inputs = batch[:, :-1]
    targets = batch[:, 1:].clone()
    equals_positions = (inputs == TOKEN_TO_ID["="]).nonzero(as_tuple=False)
    for row, column in equals_positions:
        targets[row, :column] = IGNORE_INDEX
    return inputs, targets


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=IGNORE_INDEX)


@torch.inference_mode()
def complete(model: torch.nn.Module, prompt: str, device: torch.device) -> str:
    token_ids = encode(prompt)
    while len(token_ids) < CONTEXT_LENGTH:
        inputs = torch.tensor(token_ids, device=device, dtype=torch.long).unsqueeze(0)
        logits = model(inputs)
        token_ids.append(int(logits[0, -1].argmax()))
    return decode(token_ids)
