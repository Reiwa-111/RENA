from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from .config import DEFAULT_CONFIG, ROOT, load_config, pick_device, resolve_path
from .data import load_datasets
from .engine import Trainer
from .generate import extract_reply, first_line, generate_text, load_for_inference
from .model import RENAForCausalLM
from .tokenizer import load_tokenizer, prepare_data

DEFAULT_RUN_DIR = ROOT / "runs" / "rena_rune"


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def find_checkpoint(value: str | None) -> Path:
    if value:
        path = Path(value)
        if not path.is_absolute():
            path = ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path

    for name in ("best.pt", "last.pt"):
        path = DEFAULT_RUN_DIR / name
        if path.exists():
            return path
    raise FileNotFoundError(
        f"no checkpoint in {DEFAULT_RUN_DIR}. Run: python RENA_RUNE.py train"
    )


def print_config(cfg: dict) -> None:
    m = cfg["model"]
    t = cfg["training"]
    print(f"project:   {cfg['project']['name']} v{cfg['project']['version']}")
    print(f"profile:   {cfg['project'].get('profile', '-')}")
    print(f"model:     {m['n_layers']}L d_model={m['d_model']} "
          f"heads={m['n_heads']}/{m['n_kv_heads']} head_dim={m['head_dim']} "
          f"d_ff={m['d_ff']} ctx={m['context_size']} vocab={m['vocab_size']}")
    print(f"attention: {cfg['attention']['pattern']} | norm={m['norm']} "
          f"| qk_norm={m.get('qk_norm')} | rope_theta={m.get('rope_theta')}")
    print(f"training:  steps={t['steps']} batch={t['micro_batch_size']}x{t['grad_accum']} "
          f"lr={t['learning_rate']} schedule={t['lr_schedule']}")
    print(f"runtime:   device={cfg['runtime'].get('device')} "
          f"threads={cfg['runtime'].get('threads')}")
    print(f"config:    {cfg.get('_meta', {}).get('path', DEFAULT_CONFIG)}")


def build_model(cfg: dict) -> RENAForCausalLM:
    torch.manual_seed(int(cfg["project"].get("seed", 42)))
    return RENAForCausalLM(cfg)


def parameter_report(model: RENAForCausalLM) -> None:
    embed = model.embed.weight.numel()
    blocks = sum(p.numel() for p in model.blocks.parameters())
    tied_head = getattr(model.lm_head, "embedding", None) is model.embed
    head = 0 if tied_head else sum(p.numel() for p in model.lm_head.parameters())
    norm = sum(p.numel() for p in model.norm.parameters())
    total = model.count_parameters()
    print(f"total parameters: {total:,}")
    print(f"  embeddings: {embed:,} ({embed / total:.1%})")
    print(f"  blocks:     {blocks:,} ({blocks / total:.1%})")
    print(f"  lm_head:    {head:,} (tied)" if tied_head else f"  lm_head:    {head:,}")
    print(f"  final norm: {norm:,}")
    print(f"  per layer:  {blocks / max(1, len(model.blocks)):,.0f}")


# ----------------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------------

def cmd_validate(args) -> None:
    cfg = load_config(args.config)
    print_config(cfg)
    print("config valid")


def cmd_inspect(args) -> None:
    cfg = load_config(args.config)
    model = build_model(cfg)
    print_config(cfg)
    parameter_report(model)


def cmd_tokenizer(args) -> None:
    cfg = load_config(args.config)
    if args.action == "train":
        meta = prepare_data(cfg, force=True, retrain_tokenizer=True)
        print(json.dumps(meta, ensure_ascii=False, indent=2))
        return

    path = resolve_path(cfg, cfg["tokenizer"]["path"])
    tok = load_tokenizer(path)
    print(f"tokenizer: {path}")
    print(f"vocab: {tok.get_vocab_size():,}")
    samples = [
        'Susie: "Hey, Kris."',
        "[Scene: Obj Classroom]",
        'Ralsei: "We have to save the world!"',
    ]
    for sample in samples:
        ids = tok.encode(sample).ids
        restored = tok.decode(ids, skip_special_tokens=True)
        ok = restored == sample
        print(f"  {sample!r} -> {len(ids)} tokens | roundtrip {'ok' if ok else 'MISMATCH'}")
    corpus = resolve_path(cfg, cfg["data"]["corpus_path"])
    if corpus.exists():
        text = corpus.read_text(encoding="utf-8")
        ids = tok.encode(text, add_special_tokens=False).ids
        print(f"corpus: {len(text):,} chars -> {len(ids):,} tokens "
              f"({len(text) / max(1, len(ids)):.2f} chars/token)")


def cmd_prepare(args) -> None:
    cfg = load_config(args.config)
    meta = prepare_data(cfg, force=args.force)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def cmd_train(args) -> None:
    cfg = load_config(args.config)
    if args.steps:
        cfg["training"]["steps"] = int(args.steps)
        cfg["training"]["schedule_steps"] = int(args.steps)
    if args.lr:
        cfg["training"]["learning_rate"] = float(args.lr)
    if args.batch_size:
        cfg["training"]["micro_batch_size"] = int(args.batch_size)
    if args.threads:
        cfg["runtime"]["threads"] = int(args.threads)
    if args.device:
        cfg["runtime"]["device"] = args.device

    prepare_data(cfg, force=args.force_prepare)
    train_data, val_data = load_datasets(cfg)
    tokenizer = load_tokenizer(resolve_path(cfg, cfg["tokenizer"]["path"]))

    model = build_model(cfg)
    run_dir = Path(args.run_dir) if args.run_dir else DEFAULT_RUN_DIR
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir

    device = pick_device(cfg)
    trainer = Trainer(
        model=model,
        cfg=cfg,
        train_data=train_data,
        val_data=val_data,
        tokenizer=tokenizer,
        run_dir=run_dir,
        device=str(device),
        threads=int(cfg["runtime"].get("threads", 6)),
        total_steps=int(cfg["training"]["steps"]),
    )

    print_config(cfg)
    parameter_report(model)
    if device.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)} | "
              f"torch {torch.__version__} | hip {torch.version.hip} | "
              f"amp {trainer.amp} ({trainer.amp_dtype})")
    else:
        print(f"device: cpu | torch {torch.__version__}")
    print(f"data:  train {len(train_data):,} tokens | val {len(val_data):,} tokens")
    print(f"run:   {run_dir}")
    print()

    resume = args.resume
    if resume == "auto":
        candidate = run_dir / "last.pt"
        resume = str(candidate) if candidate.exists() else None

    result = trainer.train(resume=resume)
    print()
    print(f"finished: step {result['steps']} | best val {result['best_val_loss']:.4f} "
          f"| {result['elapsed_seconds']:.1f}s")


def cmd_sample(args) -> None:
    ckpt_path = find_checkpoint(args.checkpoint)
    cfg = load_config(args.config)
    device = pick_device(cfg)
    model, tok, ckpt = load_for_inference(
        ckpt_path, device=str(device),
        threads=int(cfg["runtime"].get("threads", 6)),
    )
    if args.seed is not None:
        torch.manual_seed(args.seed)

    print(f"[{cfg['project']['name']} | step {ckpt.get('step')} "
          f"| params {model.count_parameters():,} | best val "
          f"{ckpt.get('best_val_loss', float('nan')):.4f}]")

    text = generate_text(
        model, tok, args.prompt,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        device=str(device),
    )
    print(text)


def cmd_talk(args) -> None:
    ckpt_path = find_checkpoint(args.checkpoint)
    cfg = load_config(args.config)
    chat = cfg.get("chat", {})
    device = pick_device(cfg)
    model, tok, ckpt = load_for_inference(
        ckpt_path, device=str(device),
        threads=int(cfg["runtime"].get("threads", 6)),
    )
    speaker = args.speaker or chat.get("default_speaker", "Susie")
    user_speaker = args.as_speaker or chat.get("default_user_speaker", "Kris")
    temperature = float(args.temperature or chat.get("temperature", 0.85))
    context_lines = int(chat.get("context_lines", 6))
    max_new = int(chat.get("max_new_tokens", 120))

    history: list[str] = []
    print(f"[{cfg['project']['name']} | step {ckpt.get('step')} | params "
          f"{model.count_parameters():,}]")
    print(f"You write like a {user_speaker}, the model responds as a {speaker}.")
    print("commands: /exit, /reset, /speaker NAME, /temp X")
    print()

    while True:
        try:
            user = input(f"{user_speaker}> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            history.clear()
            print("[context cleared]")
            continue
        if user.startswith("/speaker "):
            speaker = user.split(maxsplit=1)[1].strip()
            print(f"[speaker = {speaker}]")
            continue
        if user.startswith("/temp "):
            try:
                temperature = float(user.split(maxsplit=1)[1])
                print(f"[temperature = {temperature}]")
            except ValueError:
                print("[usage: /temp 0.8]")
            continue

        history.append(f'{user_speaker}: "{user}"')
        prompt = "\n".join(history[-context_lines:]) + f'\n{speaker}: "'
        full = generate_text(
            model, tok, prompt,
            max_new_tokens=max_new,
            temperature=temperature,
            top_k=chat.get("top_k", 40),
            top_p=chat.get("top_p", 0.95),
            repetition_penalty=float(chat.get("repetition_penalty", 1.1)),
            no_repeat_ngram_size=int(chat.get("no_repeat_ngram_size", 0)),
            device=str(device),
        )
        continuation = full[len(prompt):] if full.startswith(prompt) else full
        reply = first_line(continuation).strip()
        if reply.startswith(f"{speaker}:"):
            reply = reply.split(":", 1)[1].strip()
        reply = reply.strip('"').strip()
        if not reply:
            reply = "..."
        print(f"{speaker}: {reply}")
        history.append(f'{speaker}: "{reply}"')
        print()


def cmd_smoke(args) -> None:
    cfg = load_config(args.config)
    torch.manual_seed(0)
    model = build_model(cfg)
    device = pick_device(cfg)
    model.to(device)
    m = cfg["model"]
    x = torch.randint(0, int(m["vocab_size"]), (2, min(16, int(m["context_size"]))), device=device)
    y = torch.randint(0, int(m["vocab_size"]), x.shape, device=device)

    logits, loss, _ = model(x, y)
    loss.backward()
    grad_norm = sum(
        float(p.grad.norm()) for p in model.parameters() if p.grad is not None
    )
    print(f"forward ok: logits {tuple(logits.shape)} | loss {loss.item():.4f}")
    print(f"backward ok: grad norm {grad_norm:.4f}")
    model.eval()
    idx = torch.tensor([[1, 2, 3]], dtype=torch.long, device=device)
    out = model.generate(idx, 8, temperature=1.0, top_k=10, top_p=1.0)
    print(f"generate ok: {out.shape[1]} tokens")
    print("smoke passed")


# ----------------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------------

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="RENA_RUNE.py",
        description="RENA-RUNE: mini GQA/SwiGLU LM for the Deltarune transcript",
    )
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to config JSON")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("validate-config", help="validate config and print a summary")

    sub.add_parser("inspect-model", help="build the model and print parameter counts")

    tok = sub.add_parser("tokenizer", help="BPE tokenizer utilities")
    tok.add_argument("action", choices=["train", "check"])

    prep = sub.add_parser("prepare", help="build corpus, tokenizer and token files")
    prep.add_argument("--force", action="store_true")

    train = sub.add_parser("train", help="train the model")
    train.add_argument("--steps", type=int)
    train.add_argument("--lr", type=float)
    train.add_argument("--batch-size", type=int)
    train.add_argument("--device")
    train.add_argument("--threads", type=int)
    train.add_argument("--run-dir")
    train.add_argument("--resume", help="checkpoint path or 'auto'")
    train.add_argument("--force-prepare", action="store_true")

    sample = sub.add_parser("sample", help="generate text from a checkpoint")
    sample.add_argument("--checkpoint")
    sample.add_argument("--prompt", default='Susie: "')
    sample.add_argument("--tokens", type=int, default=200)
    sample.add_argument("--temperature", type=float, default=0.85)
    sample.add_argument("--top-k", type=int, default=40)
    sample.add_argument("--top-p", type=float, default=0.95)
    sample.add_argument("--repetition-penalty", type=float, default=1.1)
    sample.add_argument("--no-repeat-ngram-size", type=int, default=0)
    sample.add_argument("--seed", type=int)

    talk = sub.add_parser("talk", help="interactive dialogue by speaker")
    talk.add_argument("--checkpoint")
    talk.add_argument("--speaker", default=None)
    talk.add_argument("--as-speaker", default=None)
    talk.add_argument("--temperature", type=float, default=None)

    sub.add_parser("smoke", help="forward/backward/generate sanity check")
    return p


def main(argv: list[str] | None = None) -> None:
    parser = make_parser()
    args = parser.parse_args(argv)
    handlers = {
        "validate-config": cmd_validate,
        "inspect-model": cmd_inspect,
        "tokenizer": cmd_tokenizer,
        "prepare": cmd_prepare,
        "train": cmd_train,
        "sample": cmd_sample,
        "talk": cmd_talk,
        "smoke": cmd_smoke,
    }
    try:
        handlers[args.command](args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
