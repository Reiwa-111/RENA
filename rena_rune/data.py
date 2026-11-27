from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import resolve_path


class TokenDataset:
    """In-memory uint16 token stream with random contiguous windows."""

    def __init__(self, path: Path, name: str = "data"):
        if not path.exists():
            raise FileNotFoundError(f"{name} token file not found: {path}")
        self.path = path
        self.name = name
        self.tokens = torch.from_numpy(
            np.fromfile(path, dtype=np.uint16).astype(np.int64)
        )

    def __len__(self) -> int:
        return int(self.tokens.numel())

    def get_batch(
        self,
        context_size: int,
        batch_size: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self) <= context_size + 1:
            raise ValueError(
                f"{self.name} split has {len(self)} tokens, need > {context_size + 1}"
            )
        max_start = len(self) - context_size - 1
        starts = torch.randint(
            0, max_start, (batch_size,), generator=generator
        )
        x = torch.stack([self.tokens[i: i + context_size] for i in starts])
        y = torch.stack([self.tokens[i + 1: i + 1 + context_size] for i in starts])
        return x.to(device), y.to(device)


def load_datasets(cfg: dict[str, Any]) -> tuple[TokenDataset, TokenDataset]:
    train = TokenDataset(resolve_path(cfg, cfg["data"]["train_tokens"]), "train")
    val = TokenDataset(resolve_path(cfg, cfg["data"]["val_tokens"]), "val")
    return train, val
