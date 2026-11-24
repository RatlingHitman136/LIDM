"""LIDM: frozen two-pass ModernBERT encoder + learned layer mix + looped fusion head (docs/ARCHITECTURE.md).

Pass A: `[CLS] state [SEP] question [SEP]`, per request.
Pass B: `[CLS] [MASK] option [SEP]`, one sequence per distinct option (cacheable at inference); the [MASK]
output is the option's marker.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from cls.head import FusionHead

ENCODER = "answerdotai/ModernBERT-large"
MARKER_POS = 1  # [MASK] right after [CLS]


def option_texts(item: dict) -> list[str]:
    return ["no", "yes"] if item["type"] == "noul" else list(item["criteria"])


class Encoder:
    """Encoder returning every layer's output as a tuple of L tensors (B x T x d, bf16). Frozen by default;
    with lora_rank > 0, LoRA (alpha 2r, dropout 0.05) on Wqkv, Wo, Wi and the MLP Wo of every layer, shared by
    both passes (design doc stage 2b settings). Not an nn.Module on purpose: the frozen weights stay out of the
    trainable state_dict and the checkpoints; only the LoRA weights are saved (lora_state / load_lora)."""

    def __init__(self, name: str = ENCODER, device: str = "cuda", max_len: int = 512, max_opt_len: int = 64,
                 lora_rank: int = 0, option_attn: str = "none", joint_layers: int = 7):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, dtype=torch.bfloat16, attn_implementation="sdpa").to(device).eval()
        self.model.requires_grad_(False)
        self.lora_rank = lora_rank
        if lora_rank:
            from peft import LoraConfig, get_peft_model

            self.model = get_peft_model(self.model, LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank,
                                                               lora_dropout=0.05, target_modules=["Wqkv", "Wo", "Wi"]))
            for p in self.lora_params():
                p.data = p.data.float()  # fp32 master weights; the frozen base stays bf16
            # recompute each layer in backward: grads through 28 layers of both passes would not fit otherwise
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.device, self.max_len, self.max_opt_len = device, max_len, max_opt_len
        cfg = self.model.config
        self.d, self.heads, self.inter, self.n_layers = (cfg.hidden_size, cfg.num_attention_heads,
                                                         cfg.intermediate_size, cfg.num_hidden_layers)
        self.pooling = _pooling(name)
        self.marker = True  # pass B prepends the [MASK] marker; False for option_vector 'cls' / 'mean'
        assert option_attn in ("none", "state", "dj"), option_attn
        # 'state' = LIDM-X: options read pass A at every layer; 'dj' = LIDM-DJ: the top joint_layers run each
        # option jointly with its question's pass A (both directions); both through encode_x
        self.option_attn, self.joint_layers = option_attn, joint_layers

    @torch.no_grad()
    def embed(self, texts: list[str], max_length: int = 512):
        """The encoder's own retrieval embedding (`[CLS] text [SEP]`, its sentence-transformers pooling,
        L2-normalised), with LoRA switched off so training cannot wear down the pretrained geometry."""
        import contextlib

        enc = self.tok(list(texts), padding=True, truncation=True, max_length=max_length,
                       return_tensors="pt").to(self.device)
        with self.model.disable_adapter() if self.lora_rank else contextlib.nullcontext():
            h = self.model(**enc).last_hidden_state.float()
        v = h[:, 0] if self.pooling == "cls" else _masked_mean(h, enc["attention_mask"].bool())
        return F.normalize(v, dim=-1).cpu().numpy()

    def lora_params(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def lora_state(self) -> dict:
        return {n: p.detach().cpu() for n, p in self.model.named_parameters() if p.requires_grad}

    def load_lora(self, state: dict) -> None:
        missing = set(state) - {n for n, _ in self.model.named_parameters()}
        assert not missing, f"LoRA weights not in this encoder: {sorted(missing)[:3]}"
        self.model.load_state_dict(state, strict=False)

    def train(self, mode: bool = True) -> None:
        """Training mode only matters with LoRA (its dropout, gradient checkpointing); ModernBERT's own dropouts are 0."""
        self.model.train(mode and bool(self.lora_rank))

    def _layers(self, enc) -> tuple[torch.Tensor, ...]:
        # gradients only with LoRA during training (callers wrap validation in no_grad);
        # drop the embedding output; no torch.stack, which would copy all 28 layers once more
        with torch.set_grad_enabled(bool(self.lora_rank) and torch.is_grad_enabled()):
            return self.model(**enc, output_hidden_states=True).hidden_states[1:]

    def encode_a(self, states: list[str], questions: list[str]):
        enc = self.tok(states, questions, padding=True, truncation="only_first", max_length=self.max_len,
                       return_tensors="pt").to(self.device)
        return self._layers(enc), enc["attention_mask"].bool()

    def encode_options(self, texts: list[str], marker: bool = True):
        """Distinct option strings -> (L tensors U x To x d, U x To), padded to the longest, truncated at
        max_opt_len. Called per batch: caching every option of a large label set would not fit in VRAM
        (all layers cost ~57 KB per token). marker=False encodes `[CLS] option [SEP]` as retrieval models
        were trained, for scoring from the encoder's own pooling instead of a [MASK] marker."""
        texts = [f"{self.tok.mask_token} {t}" for t in texts] if marker else list(texts)
        enc = self.tok(texts, padding=True, truncation=True, max_length=self.max_opt_len,
                       return_tensors="pt").to(self.device)
        if marker:
            assert (enc["input_ids"][:, MARKER_POS] == self.tok.mask_token_id).all()
        return self._layers(enc), enc["attention_mask"].bool()


    def _base(self):
        return self.model.get_base_model() if self.lora_rank else self.model

    def encode_x(self, states: list[str], questions: list[str], options: list[list[str]]):
        """LIDM-X. Pass A as in encode_a; pass B with one row per (question, option), since an option's states now
        depend on its question: every option token also attends to its own question's pass-A keys and values at
        every layer, never to other options. Equal to one packed pass [A; O_1; ...; O_N] with that mask, every
        option starting at position len(A), so scores do not depend on option order (see _x_forward). Returns
        (pass-A layers, a_mask, option layers, option token mask, owner); owner[u] = question of option row u."""
        a = self.tok(states, questions, padding=True, truncation="only_first", max_length=self.max_len,
                     return_tensors="pt").to(self.device)
        texts = [f"{self.tok.mask_token} {t}" if self.marker else t for opts in options for t in opts]
        o = self.tok(texts, padding=True, truncation=True, max_length=self.max_opt_len, return_tensors="pt").to(self.device)
        if self.marker:
            assert (o["input_ids"][:, MARKER_POS] == self.tok.mask_token_id).all()
        owner = torch.tensor([i for i, opts in enumerate(options) for _ in opts], device=self.device)
        a_mask, o_mask = a["attention_mask"].bool(), o["attention_mask"].bool()
        with torch.set_grad_enabled(bool(self.lora_rank) and torch.is_grad_enabled()):
            if self.option_attn == "dj":
                ha, ho = _dj_forward(self._base(), a["input_ids"], a_mask, o["input_ids"], o_mask, owner, self.joint_layers)
            else:
                ha, ho = _x_forward(self._base(), a["input_ids"], a_mask, o["input_ids"], o_mask, owner)
        return ha, a_mask, ho, o_mask, owner


def _qkv_out(base):
    """ModernBERT attention pieces as functions of (layer, x): q/k/v with RoPE, and the output projection."""
    from transformers.models.modernbert.modeling_modernbert import apply_rotary_pos_emb

    heads = base.config.num_attention_heads
    hd = base.config.hidden_size // heads

    def qkv(layer, x, cos_sin):
        b, t, _ = x.shape
        q, k, v = layer.attn.Wqkv(x).view(b, t, 3, heads, hd).unbind(2)
        q, k = apply_rotary_pos_emb(q.transpose(1, 2), k.transpose(1, 2), *cos_sin)
        return q, k, v.transpose(1, 2)

    def out(layer, o):  # B x H x T x hd -> B x T x d
        return layer.attn.Wo(o.transpose(1, 2).reshape(o.shape[0], o.shape[2], -1))

    return qkv, out


def _self_masks(real, pos, w):
    """Masks for sequences that attend only within themselves: real B x T (key is a real token), pos B x T.
    Global layers: every real key. Local layers: real keys with |pos_q - pos_k| <= w; the diagonal keeps padded
    query rows finite (their outputs are never used)."""
    t, pos = real.shape[1], pos.int()  # int32: the U x T x T difference is the largest temporary here
    near = (pos[:, :, None] - pos[:, None, :]).abs() <= w
    local = (real[:, None, None, :] & near[:, None]) | torch.eye(t, dtype=torch.bool, device=real.device)
    return {"full_attention": real[:, None, None, :], "sliding_attention": local}


def _dj_forward(base, a_ids, a_mask, o_ids, o_mask, owner, joint):
    """LIDM-DJ (DeFormer-style deferred joint layers). Layers 1..L-joint: pass A and every option alone, each option
    at its own positions 0..L_o-1 (as LIDM's pass B, so lower-layer option states depend only on the option).
    Layers L-joint+1..L: each option row runs jointly with a copy of its question's pass A, [A ; option] with full
    two-way attention (the state reads the option, the option reads the state), the option continuing at positions
    S.. (S = its question's real length); copies never see other options. Pass A also continues alone through the
    top layers, for the layer mix and the head's cross-attention. Returns per-layer outputs like _x_forward."""
    from torch.utils.checkpoint import checkpoint

    cfg = base.config
    qkv, out = _qkv_out(base)
    w, dev, n_layers = cfg.sliding_window, a_ids.device, len(base.layers)
    split = n_layers - joint
    assert 0 < joint <= n_layers, joint
    (b, ta), (u, to) = a_ids.shape, o_ids.shape
    pos_a = torch.arange(ta, device=dev)[None].expand(b, -1)
    pos_o = torch.arange(to, device=dev)[None].expand(u, -1)
    pos_j = torch.cat([pos_a[owner], a_mask.sum(1)[owner][:, None] + pos_o], 1)  # U x (Ta + To)
    masks = {"a": _self_masks(a_mask, pos_a, w), "o": _self_masks(o_mask, pos_o, w),
             "j": _self_masks(torch.cat([a_mask[owner], o_mask], 1), pos_j, w)}
    h_a, h_o = base.embeddings(input_ids=a_ids), base.embeddings(input_ids=o_ids)
    rope = {(k, lt): base.rotary_emb(h_a, pos, lt) for k, pos in (("a", pos_a), ("o", pos_o), ("j", pos_j))
            for lt in set(cfg.layer_types)}

    def block(layer, h, key):  # one ModernBERT layer on sequences that attend within themselves
        lt = layer.attention_type
        q, k, v = qkv(layer, layer.attn_norm(h), rope[key, lt])
        h = h + out(layer, F.scaled_dot_product_attention(q, k, v, attn_mask=masks[key][lt]))
        return h + layer.mlp(layer.mlp_norm(h))

    def low(layer, h_a, h_o):
        return block(layer, h_a, "a"), block(layer, h_o, "o")

    def high(layer, h_a, h_j):
        return block(layer, h_a, "a"), block(layer, h_j, "j")

    ckpt = torch.is_grad_enabled() and base.training
    outs_a, outs_o = [], []
    for i, layer in enumerate(base.layers):
        if i == split:
            h_o = torch.cat([h_a[owner], h_o], 1)  # joint copies [A ; option] from here on
        fn = low if i < split else high
        h_a, h_o = checkpoint(fn, layer, h_a, h_o, use_reentrant=False) if ckpt else fn(layer, h_a, h_o)
        outs_a.append(h_a)
        outs_o.append(h_o if i < split else h_o[:, ta:].contiguous())  # a view would keep every joint copy alive
    outs_a[-1], outs_o[-1] = base.final_norm(h_a), base.final_norm(h_o[:, ta:])
    return tuple(outs_a), tuple(outs_o)


def _x_forward(base, a_ids, a_mask, o_ids, o_mask, owner):
    """Both passes through ModernBERT's own layers (+ LoRA), layer by layer: pass A attends to itself only, so its
    keys/values do not depend on the options; each option row attends to [its question's pass-A keys/values at that
    layer ; its own tokens]. Positions: pass A 0..S-1, every option S..S+L-1 (S = its question's real length), and
    local layers keep HF's bidirectional window |pos_q - pos_k| <= sliding_window on those positions. Returns the 28
    per-layer outputs of each pass, as Encoder._layers does (embedding output dropped, final norm on the last)."""
    from torch.utils.checkpoint import checkpoint

    cfg = base.config
    qkv, out = _qkv_out(base)
    w, dev = cfg.sliding_window, a_ids.device
    ta, to = a_ids.shape[1], o_ids.shape[1]
    pos_a = torch.arange(ta, device=dev)
    pos_o = a_mask.sum(1)[owner][:, None] + torch.arange(to, device=dev)  # U x To
    # True = may attend; the diagonal only keeps padded query rows finite (their outputs are never used)
    full_a = a_mask[:, None, None, :] | torch.eye(ta, dtype=torch.bool, device=dev)
    local_a = full_a & ((pos_a[:, None] - pos_a[None]).abs() <= w)
    to_a = a_mask[owner][:, None, None, :]
    to_o = o_mask[:, None, None, :] | torch.eye(to, dtype=torch.bool, device=dev)
    near_a = ((pos_o[:, :, None] - pos_a[None, None]).abs() <= w)[:, None]
    near_o = ((pos_o[:, :, None] - pos_o[:, None]).abs() <= w)[:, None]
    masks = {"full_attention": (full_a, torch.cat([to_a.expand(-1, -1, to, -1), to_o], -1)),
             "sliding_attention": (local_a, torch.cat([to_a & near_a, to_o & near_o], -1))}
    h_a, h_o = base.embeddings(input_ids=a_ids), base.embeddings(input_ids=o_ids)
    rope = {lt: (base.rotary_emb(h_a, pos_a[None].expand(len(a_ids), -1), lt), base.rotary_emb(h_o, pos_o, lt))
            for lt in set(cfg.layer_types)}

    def step(layer, h_a, h_o):
        (r_a, r_o), (m_a, m_o) = rope[layer.attention_type], masks[layer.attention_type]
        q_a, k_a, v_a = qkv(layer, layer.attn_norm(h_a), r_a)
        q_o, k_o, v_o = qkv(layer, layer.attn_norm(h_o), r_o)
        att_o = F.scaled_dot_product_attention(q_o, torch.cat([k_a[owner], k_o], 2), torch.cat([v_a[owner], v_o], 2),
                                               attn_mask=m_o)
        h_a = h_a + out(layer, F.scaled_dot_product_attention(q_a, k_a, v_a, attn_mask=m_a))
        h_o = h_o + out(layer, att_o)
        return h_a + layer.mlp(layer.mlp_norm(h_a)), h_o + layer.mlp(layer.mlp_norm(h_o))

    ckpt = torch.is_grad_enabled() and base.training  # recompute each layer in backward, as HF does with LoRA
    outs_a, outs_o = [], []
    for layer in base.layers:
        h_a, h_o = checkpoint(step, layer, h_a, h_o, use_reentrant=False) if ckpt else step(layer, h_a, h_o)
        outs_a.append(h_a)
        outs_o.append(h_o)
    outs_a[-1], outs_o[-1] = base.final_norm(h_a), base.final_norm(h_o)  # HF's last hidden state is after final_norm
    return tuple(outs_a), tuple(outs_o)


def _pooling(name: str) -> str:
    """'cls' or 'mean', from the model's sentence-transformers pooling config; 'mean' for plain MLM checkpoints."""
    import json

    from huggingface_hub import hf_hub_download
    try:
        path = hf_hub_download(name, "1_Pooling/config.json")
    except Exception:  # noqa: BLE001 - any failure here means no pooling config: not a sentence-embedding model
        return "mean"
    with open(path) as f:
        cfg = json.load(f)
    return "cls" if cfg.get("pooling_mode_cls_token") else "mean"


def _masked_mean(x, m):
    m = m.unsqueeze(-1).to(x.dtype)
    return (x * m).sum(-2) / m.sum(-2).clamp(min=1)


class LIDM(nn.Module):
    def __init__(self, d=1024, heads=16, inter=2624, n_layers=28, n_blocks=2, loops=1, lora_rank=16,
                 dual_score=True, option_vector="marker"):
        """option_vector: how an option's head output becomes its score: 'marker' = the [MASK] token after [CLS]
        (needs Encoder.encode_options(marker=True)); 'cls' = its first token; 'mean' = mean of its tokens."""
        super().__init__()
        assert option_vector in ("marker", "cls", "mean"), option_vector
        self.option_vector = option_vector
        self.mix_a = nn.Parameter(torch.zeros(n_layers))  # softmax layer mix per pass (Peters et al.)
        self.mix_b = nn.Parameter(torch.zeros(n_layers))
        self.head = FusionHead(d, heads, inter, n_blocks, loops, lora_rank)
        self.scorer = nn.Linear(d, 1)
        self.dual_scale = nn.Parameter(torch.tensor(1.0)) if dual_score else None

    @staticmethod
    def _mix(h, w):  # h: L tensors (... x d), as a tuple or an L x ... x d tensor; w: L
        """Softmax mix of per-layer LayerNormed states. ModernBERT's raw residual stream grows from norm ~30
        to ~1200 per token with single dims up to ~27k in the top layers; un-normalised, those dominate the
        head (attention sinks, input-independent logits). Loop over layers to avoid an L-sized fp32 copy."""
        w = w.softmax(0)
        out = 0
        for wi, hi in zip(w, h):
            out = out + wi * F.layer_norm(hi.float(), hi.shape[-1:])
        return out.to(h[0].dtype)

    def forward(self, ha, a_mask, cache_h, cache_mask, opt_idx, opt_mask, loops=None, prior=None):
        """ha: L x (Q x T x d) pass A; cache_h: L x (U x To x d) for the U distinct options, as returned by
        Encoder.encode_options; opt_idx/opt_mask: Q x N indices into U. prior: optional Q x N cosine from an
        external retrieval model (e.g. bge), used instead of the encoder cosine. Returns Q x N logits."""
        a = self._mix(ha, self.mix_a)
        o_all = self._mix(cache_h, self.mix_b)  # U x To x d; mix before gather keeps memory small
        q, n = opt_idx.shape
        to = cache_mask.shape[1]
        # Pack each question's real option tokens to the front, padded only to the longest question. A
        # Q x (N x To) layout padded every option to the longest one and every question to the most options,
        # leaving ~11% real slots on stage-1 data; the head is permutation-equivariant, so packing is exact.
        flat = (cache_mask[opt_idx] & opt_mask[..., None]).reshape(q, n * to)  # Q x (N*To), True = real token
        pos = flat.cumsum(1) - 1  # packed position of each real token
        rows, cols = flat.nonzero(as_tuple=True)
        x0 = o_all.new_zeros(q, int(flat.sum(1).max()), o_all.shape[-1])
        x0[rows, pos[rows, cols]] = o_all[opt_idx[rows, cols // to], cols % to]
        x_mask = torch.zeros(x0.shape[:2], dtype=torch.bool, device=x0.device)
        x_mask[rows, pos[rows, cols]] = True
        x = self.head(x0, x_mask, a, a_mask, loops)
        if self.option_vector == "mean":  # mean of each option's own head outputs
            seg = rows * n + cols // to  # (question, option) of each real token
            vec = x.new_zeros(q * n, x.shape[-1]).index_add_(0, seg, x[rows, pos[rows, cols]])
            vec = vec / torch.bincount(seg, minlength=q * n).clamp(min=1)[:, None].to(vec.dtype)
            vec = vec.view(q, n, -1)
        else:  # one token per option: the [MASK] marker (position 1) or [CLS] (position 0)
            slot = pos[:, (MARKER_POS if self.option_vector == "marker" else 0)::to].clamp(min=0)  # Q x N packed index
            vec = x.gather(1, slot[..., None].expand(-1, -1, x.shape[-1]))
        logits = self.scorer(vec).squeeze(-1).float()
        if self.dual_scale is not None and prior is not None:
            logits = logits + self.dual_scale * prior
        elif self.dual_scale is not None:  # retrieval prior: cos(pooled pass A, pooled option)
            o_mean = _masked_mean(o_all, cache_mask)[opt_idx]  # Q x N x d
            cos = F.cosine_similarity(_masked_mean(a, a_mask)[:, None].float(), o_mean.float(), dim=-1)
            logits = logits + self.dual_scale * cos
        return logits.masked_fill(~opt_mask, float("-inf"))


def decision_loss(logits, gold, opt_mask, is_score, rps_weight: float = 0.5, target=None):
    """Cross-entropy per question (= KL to the target up to a constant) + rps_weight x RPS for score questions.
    target: optional Q x N distributions (soft labels, zero on padded options); default one-hot at gold."""
    logp = logits.log_softmax(-1)
    if target is None:
        ce = -logp.gather(1, gold[:, None]).squeeze(1)
        cum_t = (torch.arange(logits.shape[1], device=logits.device)[None] >= gold[:, None]).float()
    else:
        ce = -(target * logp.masked_fill(~opt_mask, 0.0)).sum(-1)  # padded options: -inf logits, 0 target
        cum_t = target.cumsum(-1)
    cum_p = logp.exp().cumsum(-1)
    k = opt_mask.sum(-1).clamp(min=2)
    rps = ((cum_p - cum_t) ** 2 * opt_mask).sum(-1) / (k - 1)
    return (ce + rps_weight * rps * is_score.float()).mean()
