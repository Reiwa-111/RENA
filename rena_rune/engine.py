from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

from .config import resolve_path
from .data import TokenDataset
from .model import RENAForCausalLM


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class WarmupCosine:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        min_lr: float,
        warmup_steps: int,
        schedule_steps: int,
        mode: str = "cosine",
    ):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.min_lr = float(min_lr)
        self.warmup_steps = max(0, int(warmup_steps))
        self.schedule_steps = max(1, int(schedule_steps))
        self.mode = str(mode).lower()
        if self.mode not in {"cosine", "linear", "constant"}:
            raise ValueError(f"unsupported lr schedule: {mode}")

    def lr_at(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        if step >= self.schedule_steps:
            return self.min_lr
        if self.mode == "constant":
            return self.base_lr

        progress = (step - self.warmup_steps) / max(1, self.schedule_steps - self.warmup_steps)
        if self.mode == "linear":
            coeff = 1.0 - progress
        else:
            coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * coeff

    def set_step(self, step: int) -> float:
        lr = self.lr_at(step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


class Trainer:
    def __init__(
        self,
        model: RENAForCausalLM,
        cfg: dict[str, Any],
        train_data: TokenDataset,
        val_data: TokenDataset,
        tokenizer,
        run_dir: str | Path,
        device: str | torch.device = "cpu",
        threads: int | None = None,
        total_steps: int | None = None,
    ):
        self.cfg = cfg
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.tokenizer = tokenizer
        if str(device).lower() == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)

        if threads and self.device.type == "cpu":
            torch.set_num_threads(int(threads))

        runtime = cfg.get("runtime", {})
        precision = str(runtime.get("matmul_precision", "high")).lower()
        if precision in {"highest", "high", "medium"}:
            torch.set_float32_matmul_precision(precision)

        amp_dtype = str(runtime.get("amp_dtype", "bfloat16")).lower()
        self.amp = bool(runtime.get("amp", False)) and self.device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16 if amp_dtype in {"bf16", "bfloat16"} else torch.float16
        )
        if self.amp and self.amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.float16

        self.model.to(self.device)

        t = cfg["training"]
        self.total_steps = int(total_steps or t["steps"])
        self.micro_batch_size = int(t["micro_batch_size"])
        self.grad_accum = int(t["grad_accum"])
        self.context_size = int(cfg["model"]["context_size"])
        self.log_every = int(t.get("log_every", 25))
        self.validate_every = int(t.get("validate_every", 250))
        self.save_every = int(t.get("save_every", 500))
        self.sample_every = int(t.get("sample_every", 500))
        self.val_batches = int(t.get("val_batches", 20))
        self.grad_clip = float(t.get("grad_clip", 1.0))
        self.early_stop_patience = int(t.get("early_stop_patience", 0))
        self.early_stop_min_delta = float(t.get("early_stop_min_delta", 0.0))

        decay = [p for p in model.parameters() if p.dim() >= 2]
        no_decay = [p for p in model.parameters() if p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": float(t.get("weight_decay", 0.01))},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=float(t["learning_rate"]),
            betas=tuple(t.get("betas", (0.9, 0.95))),
            eps=float(t.get("eps", 1e-8)),
        )

        warmup = min(int(t.get("warmup_steps", 0)), max(1, self.total_steps // 20))
        self.scheduler = WarmupCosine(
            self.optimizer,
            base_lr=float(t["learning_rate"]),
            min_lr=float(t.get("min_lr", float(t["learning_rate"]) * 0.1)),
            warmup_steps=warmup,
            schedule_steps=int(t.get("schedule_steps", self.total_steps)),
            mode=str(t.get("lr_schedule", "cosine")),
        )

        self.step = 0
        self.best_val_loss = float("inf")
        self.bad_evals = 0
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.sample_prompt = "Susie: \""
        self._build_val_cache()

        self._write_run_config()
        self._write_metrics({
            "event": "start",
            "step": 0,
            "params": model.count_parameters(),
            "train_tokens": len(train_data),
            "val_tokens": len(val_data),
            "device": str(self.device),
            "device_name": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "cpu",
            "amp": self.amp,
            "amp_dtype": str(self.amp_dtype),
            "config_hash": cfg.get("_meta", {}).get("hash", ""),
        })

    def _forward_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.amp:
            with torch.autocast(device_type="cuda", dtype=self.amp_dtype):
                _, loss, _ = self.model(x, y)
        else:
            _, loss, _ = self.model(x, y)
        return loss

    def _build_val_cache(self) -> None:
        generator = torch.Generator().manual_seed(int(self.cfg["project"].get("seed", 42)))
        self.val_cache = [
            self.val_data.get_batch(
                self.context_size,
                self.micro_batch_size,
                self.device,
                generator=generator,
            )
            for _ in range(self.val_batches)
        ]

    def _write_run_config(self) -> None:
        path = self.run_dir / "config.json"
        path.write_text(
            json.dumps(self.cfg, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def _write_metrics(self, record: dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("time", time.strftime("%Y-%m-%dT%H:%M:%S"))
        with self.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @torch.no_grad()
    def evaluate(self) -> float:
        was_training = self.model.training
        self.model.eval()
        total = 0.0
        for x, y in self.val_cache:
            loss = self._forward_loss(x, y)
            total += float(loss.item())
        if was_training:
            self.model.train()
        return total / max(1, len(self.val_cache))

    def save_checkpoint(self, path: Path, event: str = "checkpoint") -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "config": self.cfg,
            "tokenizer": {
                "path": str(resolve_path(self.cfg, self.cfg["tokenizer"]["path"])),
                "vocab_size": self.tokenizer.get_vocab_size(),
            },
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
        }
        torch.save(payload, tmp)
        os.replace(tmp, path)
        self._write_metrics({
            "event": event,
            "step": self.step,
            "val_loss": self.best_val_loss,
            "checkpoint": str(path),
        })

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        self.step = int(ckpt.get("step", 0))
        self.best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        if "torch_rng_state" in ckpt:
            torch.set_rng_state(ckpt["torch_rng_state"])
        if self.device.type == "cuda" and ckpt.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(ckpt["cuda_rng_state"])
        print(f"resumed {path} at step {self.step} (best val {self.best_val_loss:.4f})")

    def _log_step(self, step: int, loss: float, lr: float, elapsed: float, window_steps: int) -> None:
        it_s = window_steps / max(elapsed, 1e-9)
        eta = (self.total_steps - step - 1) / max(it_s, 1e-9)
        tokens_seen = (step + 1) * self.micro_batch_size * self.grad_accum * self.context_size
        print(
            f"step {step + 1:6d}/{self.total_steps} | loss {loss:.4f} "
            f"| lr {lr:.2e} | {it_s:.2f} it/s | tokens {tokens_seen / 1e6:.2f}M "
            f"| eta {format_duration(eta)}"
        )
        self._write_metrics({
            "event": "train",
            "step": step + 1,
            "loss": loss,
            "lr": lr,
            "it_per_s": it_s,
            "tokens_seen": tokens_seen,
        })

    def _sample(self) -> str:
        ids = self.tokenizer.encode(self.sample_prompt).ids or [0]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            idx,
            max_new_tokens=90,
            temperature=0.8,
            top_k=40,
            top_p=0.95,
        )
        text = self.tokenizer.decode(out[0].tolist())
        return text.split("\n")[0]

    def train(self, resume: str | Path | None = None) -> dict[str, Any]:
        if resume:
            self.load_checkpoint(resume)

        self.model.train()
        run_start = time.time()
        window_start = run_start
        window_loss = 0.0
        window_steps = 0
        last_loss = float("nan")

        for step in range(self.step, self.total_steps):
            lr = self.scheduler.set_step(step)
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for _ in range(self.grad_accum):
                x, y = self.train_data.get_batch(
                    self.context_size,
                    self.micro_batch_size,
                    self.device,
                )
                loss = self._forward_loss(x, y)
                (loss / self.grad_accum).backward()
                step_loss += float(loss.item()) / self.grad_accum

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.step = step + 1
            last_loss = step_loss
            window_loss += step_loss
            window_steps += 1

            if self.step % self.log_every == 0 or self.step == self.total_steps:
                now = time.time()
                avg = window_loss / max(1, window_steps)
                self._log_step(step, avg, lr, now - window_start, window_steps)
                window_start = now
                window_loss = 0.0
                window_steps = 0

            if self.step % self.validate_every == 0 or self.step == self.total_steps:
                val_loss = self.evaluate()
                improved = val_loss < self.best_val_loss - self.early_stop_min_delta
                if improved:
                    self.best_val_loss = val_loss
                    self.bad_evals = 0
                else:
                    self.bad_evals += 1

                marker = " *best*" if improved else ""
                print(f"  eval step {self.step}: val loss {val_loss:.4f}{marker}")
                self._write_metrics({
                    "event": "eval",
                    "step": self.step,
                    "val_loss": val_loss,
                    "best_val_loss": self.best_val_loss,
                })

                if improved:
                    self.save_checkpoint(self.run_dir / "best.pt", event="best")

                if self.early_stop_patience > 0 and self.bad_evals >= self.early_stop_patience:
                    print(
                        f"early stop: no val improvement for {self.bad_evals} evals "
                        f"(patience {self.early_stop_patience})"
                    )
                    self.save_checkpoint(self.run_dir / "last.pt", event="early_stop")
                    break

            if self.sample_every and self.step % self.sample_every == 0:
                try:
                    print("  sample:", repr(self._sample()))
                except Exception as exc:  # sampling must never kill training
                    print(f"  sample failed: {exc}")
                self.model.train()

            if self.step % self.save_every == 0:
                self.save_checkpoint(self.run_dir / "last.pt", event="checkpoint")

        self.save_checkpoint(self.run_dir / "last.pt", event="finish")
        elapsed = time.time() - run_start
        result = {
            "steps": self.step,
            "best_val_loss": self.best_val_loss,
            "last_loss": last_loss,
            "elapsed_seconds": elapsed,
        }
        self._write_metrics({"event": "finish", **result})
        return result
