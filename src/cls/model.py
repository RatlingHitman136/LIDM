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
                 lora_rank: int = 0):
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


def decision_loss(logits, gold, opt_mask, is_score, rps_weight: float = 0.5):
    """Cross-entropy per question (= KL to a one-hot target) + rps_weight x RPS for score questions."""
    logp = logits.log_softmax(-1)
    ce = -logp.gather(1, gold[:, None]).squeeze(1)
    cum_p = logp.exp().cumsum(-1)
    cum_t = (torch.arange(logits.shape[1], device=logits.device)[None] >= gold[:, None]).float()
    k = opt_mask.sum(-1).clamp(min=2)
    rps = ((cum_p - cum_t) ** 2 * opt_mask).sum(-1) / (k - 1)
    return (ce + rps_weight * rps * is_score.float()).mean()
