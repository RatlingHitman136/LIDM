"""Looped fusion head (docs/ARCHITECTURE.md): option tokens self-attend within their question and
cross-attend to pass-A tokens; GeGLU FFN; sandwich norm; input injection; per-loop FFN LoRA.

No positions are added across options, so scores are permutation-equivariant in the options.
Layer names (Wqkv, Wo, Wi) match ModernBERT so blocks can be initialised from its top layers.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.d, self.heads = d, heads
        self.Wqkv = nn.Linear(d, 3 * d, bias=False)
        self.Wo = nn.Linear(d, d, bias=False)

    def forward(self, x, ctx, ctx_mask):
        """x: B x Tq x d, ctx: B x Tk x d, ctx_mask: B x Tk (True = attend)."""
        w = self.Wqkv.weight
        q = F.linear(x, w[: self.d])
        k, v = F.linear(ctx, w[self.d:]).chunk(2, dim=-1)
        b, tq, h = x.shape[0], x.shape[1], self.heads
        q, k, v = (t.view(b, -1, h, self.d // h).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=ctx_mask[:, None, None, :])
        return self.Wo(o.transpose(1, 2).reshape(b, tq, self.d))


class GeGLU(nn.Module):
    def __init__(self, d: int, inter: int, loops: int, rank: int):
        super().__init__()
        self.Wi = nn.Linear(d, 2 * inter, bias=False)
        self.Wo = nn.Linear(inter, d, bias=False)
        self.rank = rank
        if rank:  # per-loop LoRA; B starts at zero so every loop begins as the shared block
            self.Ai = nn.Parameter(torch.randn(loops, rank, d) / d ** 0.5)
            self.Bi = nn.Parameter(torch.zeros(loops, 2 * inter, rank))
            self.Ao = nn.Parameter(torch.randn(loops, rank, inter) / inter ** 0.5)
            self.Bo = nn.Parameter(torch.zeros(loops, d, rank))

    def forward(self, x, loop: int):
        h = self.Wi(x)
        if self.rank:
            h = h + F.linear(F.linear(x, self.Ai[loop]), self.Bi[loop])
        a, g = h.chunk(2, dim=-1)
        h = F.gelu(a) * g
        out = self.Wo(h)
        if self.rank:
            out = out + F.linear(F.linear(h, self.Ao[loop]), self.Bo[loop])
        return out


class FusionBlock(nn.Module):
    def __init__(self, d: int, heads: int, inter: int, loops: int, rank: int):
        super().__init__()
        self.self_attn = Attention(d, heads)
        self.cross_attn = Attention(d, heads)
        self.ffn = GeGLU(d, inter, loops, rank)
        # sandwich norm (pre + post per sublayer), Huginn's choice for stable looping
        self.pre = nn.ModuleList(nn.LayerNorm(d, bias=False) for _ in range(3))
        self.post = nn.ModuleList(nn.LayerNorm(d, bias=False) for _ in range(3))

    def forward(self, x, x_mask, ctx, ctx_mask, loop: int):
        h = self.pre[0](x)
        x = x + self.post[0](self.self_attn(h, h, x_mask))
        x = x + self.post[1](self.cross_attn(self.pre[1](x), ctx, ctx_mask))
        return x + self.post[2](self.ffn(self.pre[2](x), loop))


class FusionHead(nn.Module):
    def __init__(self, d=1024, heads=16, inter=2624, n_blocks=2, max_loops=1, lora_rank=0):
        super().__init__()
        self.max_loops = max_loops
        # looped heads keep every loop's activations for backward (questions x option tokens x loops: ~10 GB for 16
        # questions with 24 long options at 4 loops); recompute each block in backward instead, as the encoder does
        self.checkpoint = max_loops > 1
        rank = lora_rank if max_loops > 1 else 0
        self.blocks = nn.ModuleList(FusionBlock(d, heads, inter, max_loops, rank) for _ in range(n_blocks))
        self.norm = nn.LayerNorm(d, bias=False)

    def forward(self, x0, x_mask, ctx, ctx_mask, loops: int | None = None, grad_loops: int = 4):
        """x0: B x T x d option tokens (all options of a question), x_mask: B x T. Returns B x T x d.

        Truncated backprop (Huginn): in training only the last `grad_loops` loops keep gradients.
        """
        loops = loops or self.max_loops
        x = x0
        for k in range(loops):
            frozen = self.training and k < loops - grad_loops
            with torch.no_grad() if frozen else contextlib.nullcontext():
                if k:
                    x = x + x0  # input injection: re-add the original option encodings every loop
                for b in self.blocks:
                    if self.checkpoint and self.training and torch.is_grad_enabled():
                        x = torch.utils.checkpoint.checkpoint(b, x, x_mask, ctx, ctx_mask, k, use_reentrant=False)
                    else:
                        x = b(x, x_mask, ctx, ctx_mask, k)
        return self.norm(x)

    @torch.no_grad()
    def init_from_encoder(self, layers) -> None:
        """Copy the encoder's top layers (MORES): self-attn weights into both attentions, MLP into FFN."""
        for blk, layer in zip(self.blocks, list(layers)[-len(self.blocks):]):
            for att in (blk.self_attn, blk.cross_attn):
                att.Wqkv.weight.copy_(layer.attn.Wqkv.weight)
                att.Wo.weight.copy_(layer.attn.Wo.weight)
            blk.ffn.Wi.weight.copy_(layer.mlp.Wi.weight)
            blk.ffn.Wo.weight.copy_(layer.mlp.Wo.weight)
            for i, src in ((0, layer.attn_norm), (2, layer.mlp_norm)):
                if isinstance(src, nn.LayerNorm):
                    blk.pre[i].weight.copy_(src.weight)
