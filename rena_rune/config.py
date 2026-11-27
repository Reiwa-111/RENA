from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config" / "rena_rune.json"


def pick_device(cfg: dict[str, Any]) -> torch.device:
    raw = str(cfg.get("runtime", {}).get("device", "auto")).lower()
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_hash(raw: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(raw).encode("utf-8")).hexdigest()


def validate_config(raw: dict[str, Any]) -> None:
    required = {"project", "model", "attention", "tokenizer", "data", "training", "runtime"}
    missing = required - set(raw)
    if missing:
        raise ValueError(f"missing config sections: {sorted(missing)}")

    m = raw["model"]
    for key in ("vocab_size", "context_size", "d_model", "n_layers",
                "n_heads", "n_kv_heads", "head_dim", "d_ff"):
        if int(m[key]) <= 0:
            raise ValueError(f"model.{key} must be > 0")

    if m["d_model"] != m["n_heads"] * m["head_dim"]:
        raise ValueError(
            "model.d_model must equal n_heads * head_dim: "
            f"{m['d_model']} != {m['n_heads']} * {m['head_dim']}"
        )
    if m["n_heads"] % m["n_kv_heads"] != 0:
        raise ValueError(
            f"n_kv_heads ({m['n_kv_heads']}) must divide n_heads ({m['n_heads']})"
        )
    if m["head_dim"] % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")
    if str(m.get("norm", "rmsnorm")).lower() not in {"rmsnorm", "layernorm", "layer_norm"}:
        raise ValueError(f"unsupported norm: {m.get('norm')}")

    a = raw["attention"]
    if str(a.get("pattern")) != "pure_gqa":
        raise ValueError("RENA-RUNE supports attention.pattern='pure_gqa'")
    if int(a["gqa"]["n_q"]) != int(m["n_heads"]):
        raise ValueError("attention.gqa.n_q must match model.n_heads")
    if int(a["gqa"]["n_kv"]) != int(m["n_kv_heads"]):
        raise ValueError("attention.gqa.n_kv must match model.n_kv_heads")

    t = raw["training"]
    if int(t["micro_batch_size"]) <= 0:
        raise ValueError("training.micro_batch_size must be > 0")
    if int(t["grad_accum"]) <= 0:
        raise ValueError("training.grad_accum must be > 0")
    if float(t["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate must be > 0")
    if int(t["schedule_steps"]) <= 0:
        raise ValueError("training.schedule_steps must be > 0")


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    raw = json.loads(path.read_text(encoding="utf-8"))
    validate_config(raw)
    # keep the hash of the file as written, not of the decorated dict
    raw["_meta"] = {"path": str(path), "hash": config_hash(raw)}
    return raw


def resolve_path(cfg: dict[str, Any], value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return ROOT / p
