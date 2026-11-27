from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.trainers import BpeTrainer

from .config import resolve_path

SPECIAL_TOKENS = ["<|pad|>", "<|bos|>", "<|eos|>"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_corpus(cfg: dict[str, Any], force: bool = False) -> Path:
    out = resolve_path(cfg, cfg["data"]["corpus_path"])
    if out.exists() and not force:
        return out

    parts = []
    for rel in cfg["data"]["corpus_files"]:
        path = resolve_path(cfg, rel)
        if not path.exists():
            raise FileNotFoundError(
                f"corpus file missing: {path}\n"
                "download the cleaned Deltarune transcript into data/"
            )
        text = path.read_text(encoding="utf-8", errors="replace")
        parts.append(text.replace("\r\n", "\n").strip())

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n\n<|chapter|>\n\n".join(parts) + "\n", encoding="utf-8")
    return out


def train_tokenizer(
    corpus_path: Path,
    out_path: Path,
    vocab_size: int = 8000,
    special_tokens: Iterable[str] = SPECIAL_TOKENS,
) -> Tokenizer:
    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus not found: {corpus_path}")

    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=int(vocab_size),
        min_frequency=2,
        special_tokens=list(special_tokens),
        show_progress=False,
        initial_alphabet=ByteLevelPreTokenizer.alphabet(),
    )
    tokenizer.train_from_iterator(
        (corpus_path.read_text(encoding="utf-8"),),
        trainer=trainer,
        length=1,
    )

    actual = tokenizer.get_vocab_size()
    if actual != int(vocab_size):
        raise RuntimeError(
            f"tokenizer produced {actual} tokens, expected {vocab_size}. "
            "Increase corpus size or lower vocab_size."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(out_path))
    return tokenizer


def load_tokenizer(path: Path) -> Tokenizer:
    if not path.exists():
        raise FileNotFoundError(
            f"tokenizer not found: {path}\n"
            "run: python RENA_RUNE.py tokenizer train"
        )
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def encode_corpus(
    cfg: dict[str, Any],
    tokenizer: Tokenizer,
    corpus_path: Path,
) -> dict[str, Any]:
    text = corpus_path.read_text(encoding="utf-8")

    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > 65535:
        raise ValueError("uint16 token files require vocab_size <= 65535")

    val_fraction = float(cfg["data"].get("val_fraction", 0.05))
    val_mode = str(cfg["data"].get("val_mode", "tail")).lower()

    if val_mode == "chunks":
        # Hold out whole line-chunks so validation keeps local dialogue context.
        chunk_lines = max(1, int(cfg["data"].get("val_chunk_lines", 20)))
        seed = int(cfg["data"].get("val_seed", 42))
        lines = text.splitlines(keepends=True)
        chunks = [lines[i: i + chunk_lines] for i in range(0, len(lines), chunk_lines)]
        order = list(range(len(chunks)))
        random.Random(seed).shuffle(order)
        n_val = max(1, int(len(chunks) * val_fraction))
        val_set = set(order[:n_val])
        train_text = "".join("".join(chunks[i]) for i in range(len(chunks)) if i not in val_set)
        val_text = "".join("".join(chunks[i]) for i in range(len(chunks)) if i in val_set)
        train_ids = tokenizer.encode(train_text, add_special_tokens=False).ids
        val_ids = tokenizer.encode(val_text, add_special_tokens=False).ids
    else:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        split = max(1, int(len(ids) * (1.0 - val_fraction)))
        train_ids, val_ids = ids[:split], ids[split:]

    import numpy as np

    train_path = resolve_path(cfg, cfg["data"]["train_tokens"])
    val_path = resolve_path(cfg, cfg["data"]["val_tokens"])
    train_path.parent.mkdir(parents=True, exist_ok=True)
    np.asarray(train_ids, dtype=np.uint16).tofile(train_path)
    np.asarray(val_ids, dtype=np.uint16).tofile(val_path)

    total_tokens = len(train_ids) + len(val_ids)
    meta = {
        "vocab_size": vocab_size,
        "corpus_chars": len(text),
        "total_tokens": total_tokens,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
        "val_fraction": val_fraction,
        "val_mode": val_mode,
        "chars_per_token": round(len(text) / max(1, total_tokens), 4),
        "tokenizer_sha256": sha256_file(resolve_path(cfg, cfg["tokenizer"]["path"])),
        "corpus_sha256": sha256_file(corpus_path),
    }
    meta_path = resolve_path(cfg, cfg["data"]["meta_path"])
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return meta


def prepare_data(cfg: dict[str, Any], force: bool = False, retrain_tokenizer: bool = False) -> dict[str, Any]:
    corpus = build_corpus(cfg, force=force)
    tok_path = resolve_path(cfg, cfg["tokenizer"]["path"])
    train_path = resolve_path(cfg, cfg["data"]["train_tokens"])
    val_path = resolve_path(cfg, cfg["data"]["val_tokens"])
    meta_path = resolve_path(cfg, cfg["data"]["meta_path"])

    if retrain_tokenizer or not tok_path.exists():
        train_tokenizer(
            corpus,
            tok_path,
            vocab_size=int(cfg["tokenizer"]["vocab_size"]),
            special_tokens=cfg["tokenizer"].get("special_tokens", SPECIAL_TOKENS),
        )

    tokenizer = load_tokenizer(tok_path)
    if tokenizer.get_vocab_size() != int(cfg["model"]["vocab_size"]):
        raise ValueError(
            f"tokenizer vocab {tokenizer.get_vocab_size()} != "
            f"model vocab {cfg['model']['vocab_size']}"
        )

    meta = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    val_mode = str(cfg["data"].get("val_mode", "tail")).lower()
    needs_encode = force or not (train_path.exists() and val_path.exists())
    if meta is not None and meta.get("val_mode") != val_mode:
        needs_encode = True

    if needs_encode:
        return encode_corpus(cfg, tokenizer, corpus)
    if meta is not None:
        return meta
    return encode_corpus(cfg, tokenizer, corpus)
