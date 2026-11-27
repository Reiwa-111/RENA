from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import json
from safetensors import safe_open
from safetensors.torch import load_file

from .config import resolve_path
from .model import RENAForCausalLM
from .tokenizer import load_tokenizer


def load_for_inference(
    checkpoint: str | Path,
    device: str = "cpu",
    threads: int = 6,
):
    torch.set_num_threads(int(threads))
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.suffix.lower() == ".safetensors":
        with safe_open(checkpoint_path, framework="pt", device="cpu") as sf:
            metadata = sf.metadata() or {}
        if metadata.get("format") != "RENA-RUNE" or "config" not in metadata:
            raise ValueError("Safetensors file is missing RENA-RUNE metadata/config")
        cfg = json.loads(metadata["config"])
        state = load_file(str(checkpoint_path), device="cpu")
        ckpt = {
            "model": state,
            "config": cfg,
            "step": int(metadata["step"]) if metadata.get("step") else None,
            "best_val_loss": float(metadata["best_val_loss"]) if metadata.get("best_val_loss") else None,
        }
    else:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = ckpt["config"]
    model = RENAForCausalLM(cfg)
    load_result = model.load_state_dict(ckpt["model"], strict=checkpoint_path.suffix.lower() != ".safetensors")
    if checkpoint_path.suffix.lower() == ".safetensors":
        allowed_missing = {"lm_head.embedding.weight"} if bool(cfg["model"].get("weight_tying", True)) else set()
        unexpected_missing = set(load_result.missing_keys) - allowed_missing
        if unexpected_missing or load_result.unexpected_keys:
            raise ValueError(
                f"Safetensors state mismatch: missing={sorted(unexpected_missing)}, "
                f"unexpected={sorted(load_result.unexpected_keys)}"
            )
    model.to(device)
    model.eval()
    tokenizer_value = Path(cfg["tokenizer"]["path"])
    if not tokenizer_value.is_absolute():
        packaged_tokenizer = Path(checkpoint).resolve().parent / tokenizer_value
        tokenizer_path = packaged_tokenizer if packaged_tokenizer.is_file() else resolve_path(cfg, str(tokenizer_value))
    else:
        tokenizer_path = tokenizer_value
    tokenizer = load_tokenizer(tokenizer_path)
    return model, tokenizer, ckpt


@torch.no_grad()
def generate_text(
    model: RENAForCausalLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.85,
    top_k: int | None = 40,
    top_p: float | None = 0.95,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
    device: str = "cpu",
) -> str:
    ids = tokenizer.encode(prompt).ids
    if not ids:
        ids = [tokenizer.token_to_id("<|bos|>") or 0]
    idx = torch.tensor([ids], dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    return tokenizer.decode(out[0].tolist(), skip_special_tokens=True)


def first_line(text: str) -> str:
    return text.split("\n", 1)[0].rstrip()


def extract_reply(line: str, prompt_prefix: str) -> str:
    reply = line[len(prompt_prefix):] if line.startswith(prompt_prefix) else line
    reply = reply.strip()
    if reply.endswith('"'):
        reply = reply[:-1].rstrip()
    return reply
