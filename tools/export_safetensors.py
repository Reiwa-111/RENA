from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import save_file

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RENA-RUNE inference weights in Safetensors format")
    parser.add_argument("checkpoint", type=Path, help="Full training checkpoint (.pt)")
    parser.add_argument("--output-dir", type=Path, default=Path("release"))
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()

    source = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    if not source.is_file():
        raise FileNotFoundError(source)

    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    tokenizer_source = Path(config["tokenizer"]["path"])
    if not tokenizer_source.is_absolute():
        tokenizer_source = ROOT / tokenizer_source
    if not tokenizer_source.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_source}")

    output.mkdir(parents=True, exist_ok=True)
    export_config = {
        "project": {"name": "RENA-RUNE", "version": str(config.get("project", {}).get("version", "0.1"))},
        "model": json.loads(json.dumps(config["model"])),
        "tokenizer": {
            "type": config.get("tokenizer", {}).get("type", "bpe_bytelevel"),
            "vocab_size": int(config["tokenizer"].get("vocab_size", config["model"]["vocab_size"])),
            "path": "tokenizer.json",
            "special_tokens": list(config.get("tokenizer", {}).get("special_tokens", [])),
        },
        "chat": json.loads(json.dumps(config.get("chat", {}))),
        "runtime": {"device": "auto", "threads": 4, "amp": False},
    }
    tied = bool(export_config["model"].get("weight_tying", True))
    tensors: dict[str, torch.Tensor] = {}
    storage_aliases: dict[tuple, torch.Tensor] = {}
    parameter_count = 0
    for name, tensor in checkpoint["model"].items():
        if tied and name.startswith("lm_head.embedding."):
            continue
        storage_key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()))
        converted = storage_aliases.get(storage_key)
        if converted is None:
            target_dtype = torch.float16 if args.dtype == "fp16" and tensor.is_floating_point() else tensor.dtype
            converted = tensor.detach().to(device="cpu", dtype=target_dtype).contiguous()
            storage_aliases[storage_key] = converted
            parameter_count += tensor.numel()
        tensors[name] = converted

    metadata = {
        "format": "RENA-RUNE",
        "format_version": "1",
        "architecture": "rena_rune",
        "config": json.dumps(export_config, separators=(",", ":"), ensure_ascii=False),
        "step": str(checkpoint.get("step", "")),
        "best_val_loss": str(checkpoint.get("best_val_loss", "")),
        "weights_dtype": args.dtype,
        "parameter_count": str(parameter_count),
    }
    weights_path = output / "model.safetensors"
    save_file(tensors, str(weights_path), metadata=metadata)
    shutil.copy2(tokenizer_source, output / "tokenizer.json")
    (output / "config.json").write_text(json.dumps(export_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    release_metadata = {k: v for k, v in metadata.items() if k != "config"}
    release_metadata.update({"weights": weights_path.name, "tokenizer": "tokenizer.json"})
    (output / "metadata.json").write_text(json.dumps(release_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Weights: {weights_path} ({weights_path.stat().st_size / (1024 ** 2):.2f} MiB)")
    print(f"Tokenizer: {output / 'tokenizer.json'}")
    print(f"Config: {output / 'config.json'}")


if __name__ == "__main__":
    main()
