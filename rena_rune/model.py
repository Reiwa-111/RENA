from __future__ import annotations

import os

os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * self.weight.float()).to(dtype)


class QKNorm(nn.Module):
    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.q = RMSNorm(head_dim, eps)
        self.k = RMSNorm(head_dim, eps)

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        return self.q(q), self.k(k)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
    """RoPE for tensors shaped [B, T, H, D]."""

    def __init__(self, dim: int, theta: float = 10000.0, max_seq_len: int = 4096):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dim must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        freqs = torch.outer(torch.arange(max_seq_len).float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
        self.max_seq_len = max_seq_len

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        t = x.shape[1]
        if offset + t > self.max_seq_len:
            raise ValueError(
                f"RoPE position {offset + t} exceeds max_seq_len {self.max_seq_len}"
            )
        cos = self.cos_cached[offset: offset + t].to(dtype=x.dtype)[None, :, None, :]
        sin = self.sin_cached[offset: offset + t].to(dtype=x.dtype)[None, :, None, :]
        return x * cos + rotate_half(x) * sin


class GQA(nn.Module):
    """Grouped-query attention with RoPE, optional QK-norm and KV cache."""

    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        m = cfg["model"]
        self.d_model = int(m["d_model"])
        self.n_heads = int(m["n_heads"])
        self.n_kv_heads = int(m["n_kv_heads"])
        self.head_dim = int(m["head_dim"])
        self.groups = self.n_heads // self.n_kv_heads
        self.dropout = float(m["dropout"])
        bias = bool(m.get("bias", False))

        self.q_proj = nn.Linear(self.d_model, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(self.d_model, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, self.d_model, bias=bias)

        self.qk_norm = (
            QKNorm(self.head_dim, float(m["norm_eps"]))
            if bool(m.get("qk_norm", False))
            else None
        )
        self.rope = RotaryEmbedding(
            self.head_dim,
            theta=float(m.get("rope_theta", 10000.0)),
            max_seq_len=int(m["context_size"]),
        )

    def _sdpa(self, q, k, v, *, causal: bool, attn_mask, dropout_p: float):
        try:
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=causal,
                enable_gqa=True,
            )
        except TypeError:
            k = k.repeat_interleave(self.groups, dim=1)
            v = v.repeat_interleave(self.groups, dim=1)
            return F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=dropout_p,
                is_causal=causal,
            )

    def forward(self, x, past=None, use_cache: bool = False):
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(b, t, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).view(b, t, self.n_kv_heads, self.head_dim)

        if self.qk_norm is not None:
            q, k = self.qk_norm(q, k)

        past_len = 0 if past is None else int(past[0].shape[-2])
        q = self.rope(q, offset=past_len)
        k = self.rope(k, offset=past_len)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if past is not None:
            k = torch.cat((past[0], k), dim=-2)
            v = torch.cat((past[1], v), dim=-2)

        attn_mask = None
        causal = past is None
        if past is not None and t > 1:
            kv_len = k.shape[-2]
            causal = False
            attn_mask = torch.ones(t, kv_len, dtype=torch.bool, device=x.device).tril(
                diagonal=kv_len - t
            )

        y = self._sdpa(
            q, k, v,
            causal=causal,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(b, t, self.n_heads * self.head_dim)
        out = self.o_proj(y)

        present = (k, v) if use_cache else None
        return out, present


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, bias: bool = False):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=bias)
        self.up = nn.Linear(d_model, d_ff, bias=bias)
        self.down = nn.Linear(d_ff, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.down(self.drop(F.silu(self.gate(x)) * self.up(x)))


class RENABlock(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        m = cfg["model"]
        d_model = int(m["d_model"])
        eps = float(m["norm_eps"])
        if str(m.get("norm", "rmsnorm")).lower() == "rmsnorm":
            self.norm1 = RMSNorm(d_model, eps)
            self.norm2 = RMSNorm(d_model, eps)
        else:
            self.norm1 = nn.LayerNorm(d_model, eps=eps)
            self.norm2 = nn.LayerNorm(d_model, eps=eps)

        self.attn = GQA(cfg)
        self.mlp = SwiGLU(
            d_model,
            int(m["d_ff"]),
            float(m["dropout"]),
            bool(m.get("bias", False)),
        )

    def forward(self, x, past=None, use_cache: bool = False):
        attn_out, present = self.attn(self.norm1(x), past=past, use_cache=use_cache)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, present


class TiedLMHead(nn.Module):
    def __init__(self, embedding: nn.Embedding):
        super().__init__()
        self.embedding = embedding

    @property
    def weight(self):
        return self.embedding.weight

    def forward(self, x):
        return F.linear(x, self.embedding.weight)


class RENAForCausalLM(nn.Module):
    def __init__(self, cfg: dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        m = cfg["model"]
        self.vocab_size = int(m["vocab_size"])
        self.context_size = int(m["context_size"])
        self.d_model = int(m["d_model"])
        self.n_layers = int(m["n_layers"])

        self.embed = nn.Embedding(self.vocab_size, self.d_model)
        self.blocks = nn.ModuleList([RENABlock(cfg) for _ in range(self.n_layers)])
        if str(m.get("norm", "rmsnorm")).lower() == "rmsnorm":
            self.norm = RMSNorm(self.d_model, float(m["norm_eps"]))
        else:
            self.norm = nn.LayerNorm(self.d_model, eps=float(m["norm_eps"]))

        self.lm_head = TiedLMHead(self.embed) if bool(m.get("weight_tying", True)) else nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.apply(self._init_weights)
        self._scale_residual_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_init(self) -> None:
        std = 0.02 / math.sqrt(2 * self.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("o_proj.weight") or name.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=std)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, input_ids, targets=None, past=None, use_cache: bool = False):
        _, t = input_ids.shape
        if t > self.context_size:
            raise ValueError(
                f"sequence length {t} exceeds context {self.context_size}"
            )

        x = self.embed(input_ids)
        present = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = None if past is None else past[i]
            x, layer_present = block(x, past=layer_past, use_cache=use_cache)
            if use_cache:
                present.append(layer_present)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                targets.reshape(-1),
            )
        return logits, loss, present

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 0.85,
        top_k: int | None = 40,
        top_p: float | None = 0.95,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
    ) -> torch.Tensor:
        was_training = self.training
        self.eval()
        try:
            if idx.shape[1] > self.context_size:
                idx = idx[:, -self.context_size:]

            logits, _, past = self(idx, use_cache=True)
            out = idx

            for _ in range(max_new_tokens):
                if out.shape[1] >= self.context_size:
                    out = out[:, -(self.context_size - 1):]
                    logits, _, past = self(out, use_cache=True)

                step_logits = logits[:, -1, :] / max(float(temperature), 1e-5)

                if repetition_penalty and repetition_penalty != 1.0:
                    score = torch.gather(step_logits, 1, out)
                    score = torch.where(
                        score > 0,
                        score / float(repetition_penalty),
                        score * float(repetition_penalty),
                    )
                    step_logits.scatter_(1, out, score)

                if no_repeat_ngram_size and no_repeat_ngram_size > 0 and out.size(1) >= no_repeat_ngram_size:
                    n = int(no_repeat_ngram_size)
                    for b in range(out.size(0)):
                        seq = out[b].tolist()
                        prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
                        banned = set()
                        for i in range(len(seq) - n + 1):
                            ngram = tuple(seq[i: i + n])
                            if ngram[:-1] == prefix:
                                banned.add(ngram[-1])
                        if banned:
                            step_logits[b, list(banned)] = -float("inf")

                if top_k is not None and 0 < top_k < step_logits.size(-1):
                    values, indices = torch.topk(step_logits, top_k, dim=-1)
                    filtered = torch.full_like(step_logits, -float("inf"))
                    filtered.scatter_(1, indices, values)
                    step_logits = filtered

                if top_p is not None and 0.0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(step_logits, descending=True, dim=-1)
                    probs = F.softmax(sorted_logits, dim=-1)
                    cumulative = torch.cumsum(probs, dim=-1)
                    remove = cumulative > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
                    step_logits = torch.full_like(step_logits, -float("inf"))
                    step_logits.scatter_(1, sorted_idx, sorted_logits)

                probs = F.softmax(step_logits, dim=-1)
                if not torch.isfinite(probs).all() or probs.sum() <= 0:
                    nxt = torch.argmax(step_logits, dim=-1, keepdim=True)
                else:
                    nxt = torch.multinomial(probs, num_samples=1)

                out = torch.cat((out, nxt), dim=1)
                logits, _, past = self(nxt, past=past, use_cache=True)

            return out
        finally:
            if was_training:
                self.train()


def model_from_config(cfg: dict[str, Any]) -> RENAForCausalLM:
    return RENAForCausalLM(cfg)
