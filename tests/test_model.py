"""CPU shape/mask/invariance checks for the M1 model with tiny dimensions (no encoder download)."""
import torch

from cls.model import LIDM, decision_loss

L, D, U, TO = 3, 32, 6, 4  # layers, width, unique options, option tokens


def setup(seed=0):
    torch.manual_seed(seed)
    m = LIDM(d=D, heads=4, inter=48, n_layers=L, n_blocks=2, loops=3, lora_rank=4)
    with torch.no_grad():  # LoRA B starts at zero; randomise so the loops really differ
        for n, p in m.named_parameters():
            if ".B" in n:
                p.normal_(0, 0.1)
    m.eval()
    ha = torch.randn(L, 2, 7, D)
    a_mask = torch.ones(2, 7, dtype=torch.bool)
    a_mask[1, 5:] = False
    cache_h = torch.randn(L, U, TO, D)  # layer-first, like Encoder.encode_options
    cache_mask = torch.ones(U, TO, dtype=torch.bool)
    cache_mask[2, 3:] = False
    return m, ha, a_mask, cache_h, cache_mask


def test_permuting_options_permutes_logits():
    m, ha, a_mask, ch, cm = setup()
    idx = torch.tensor([[0, 1, 2, 3], [4, 5, 2, 0]])
    mask = torch.ones(2, 4, dtype=torch.bool)
    perm = torch.tensor([2, 0, 3, 1])
    base = m(ha, a_mask, ch, cm, idx, mask)
    shuf = m(ha, a_mask, ch, cm, idx[:, perm], mask)
    assert torch.allclose(base[:, perm], shuf, atol=1e-5)


def test_padded_options_do_not_change_real_scores():
    m, ha, a_mask, ch, cm = setup()
    idx = torch.tensor([[0, 1, 2], [4, 5, 3]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    base = m(ha, a_mask, ch, cm, idx, mask)
    idx_pad = torch.cat([idx, torch.tensor([[5], [1]])], 1)
    mask_pad = torch.cat([mask, torch.zeros(2, 1, dtype=torch.bool)], 1)
    padded = m(ha, a_mask, ch, cm, idx_pad, mask_pad)
    assert torch.allclose(base, padded[:, :3], atol=1e-5)
    assert torch.isinf(padded[:, 3]).all()


def test_loops_change_output_and_train_with_truncated_backprop():
    m, ha, a_mask, ch, cm = setup()
    idx = torch.tensor([[0, 1, 2], [3, 4, 5]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    assert not torch.allclose(m(ha, a_mask, ch, cm, idx, mask, loops=1), m(ha, a_mask, ch, cm, idx, mask, loops=3))
    m.train()
    gold, is_score = torch.tensor([2, 1]), torch.tensor([True, False])
    loss = decision_loss(m(ha, a_mask, ch, cm, idx, mask, loops=3), gold, mask, is_score)
    loss.backward()
    assert torch.isfinite(loss) and m.mix_a.grad is not None and m.scorer.weight.grad is not None


def test_decision_loss_perfect_prediction_is_near_zero():
    logits = torch.tensor([[20.0, 0.0, 0.0], [0.0, 0.0, 20.0]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    assert decision_loss(logits, torch.tensor([0, 2]), mask, torch.tensor([False, True])) < 1e-6


def test_layer_tuples_match_stacked_tensors():
    """Encoder returns per-layer tuples; the model must give the same logits as for stacked tensors."""
    m, ha, a_mask, ch, cm = setup()
    idx = torch.tensor([[0, 1, 2], [3, 4, 5]])
    mask = torch.ones(2, 3, dtype=torch.bool)
    assert torch.allclose(m(ha, a_mask, ch, cm, idx, mask), m(tuple(ha), a_mask, tuple(ch), cm, idx, mask), atol=1e-6)


def _padded_reference(m, ha, a_mask, ch, cm, idx, mask, loops=None):
    """The pre-packing forward: every option padded to the longest, Q x (N x To) head input."""
    import torch.nn.functional as F

    from cls.model import MARKER_POS, _masked_mean
    a, o_all = m._mix(ha, m.mix_a), m._mix(ch, m.mix_b)
    to = int(cm[idx].sum(-1).max())
    o = o_all[idx][:, :, :to]
    tok = cm[idx][:, :, :to] & mask[..., None]
    q, n = idx.shape
    x = m.head(o.reshape(q, n * to, -1), tok.reshape(q, n * to), a, a_mask, loops)
    logits = m.scorer(x.view(q, n, to, -1)[:, :, MARKER_POS]).squeeze(-1)
    cos = F.cosine_similarity(_masked_mean(a, a_mask)[:, None], _masked_mean(o, tok), dim=-1)
    return (logits + m.dual_scale * cos).masked_fill(~mask, float("-inf"))


def test_packed_head_input_matches_padded_reference():
    m, ha, a_mask, ch, cm = setup()
    idx = torch.tensor([[0, 2, 1, 5], [3, 2, 4, 0]])  # option 2 is shorter (cache_mask[2, 3:] = False)
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    for loops in (1, 3):
        assert torch.allclose(m(ha, a_mask, ch, cm, idx, mask, loops=loops),
                              _padded_reference(m, ha, a_mask, ch, cm, idx, mask, loops=loops), atol=1e-5)


def test_external_prior_replaces_encoder_cosine():
    m, ha, a_mask, ch, cm = setup()
    idx = torch.tensor([[0, 1, 2], [3, 4, 5]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    p1, p2 = torch.rand(2, 3), torch.rand(2, 3)
    d = m(ha, a_mask, ch, cm, idx, mask, prior=p1) - m(ha, a_mask, ch, cm, idx, mask, prior=p2)
    assert torch.allclose(d[mask], (m.dual_scale * (p1 - p2))[mask], atol=1e-5)


def test_init_b_checkpoint_into_looped_c(tmp_path):
    """--init: a B checkpoint loads into C (per-loop LoRA missing, zero) and C at 1 loop reproduces B; a head
    with a different block count is refused."""
    import types

    import pytest

    from cls.train import load_init
    torch.manual_seed(0)
    b = LIDM(d=D, heads=4, inter=48, n_layers=L, n_blocks=2, loops=1, lora_rank=4).eval()
    path = tmp_path / "b.pt"
    torch.save({"model": b.state_dict(), "config": {}}, path)
    frozen_enc = types.SimpleNamespace(lora_rank=0)
    c = LIDM(d=D, heads=4, inter=48, n_layers=L, n_blocks=2, loops=3, lora_rank=4).eval()
    load_init(c, frozen_enc, path)
    _, ha, a_mask, ch, cm = setup()
    idx, mask = torch.tensor([[0, 1, 2], [3, 4, 5]]), torch.ones(2, 3, dtype=torch.bool)
    assert torch.allclose(b(ha, a_mask, ch, cm, idx, mask), c(ha, a_mask, ch, cm, idx, mask, loops=1), atol=1e-6)
    d = LIDM(d=D, heads=4, inter=48, n_layers=L, n_blocks=3, loops=1, lora_rank=4)
    with pytest.raises(AssertionError):
        load_init(d, frozen_enc, path)


def test_pooled_option_vectors_match_padded_reference():
    """option_vector 'cls' scores each option's first token, 'mean' the mean of its real tokens, in the packed layout."""
    from cls.model import _masked_mean
    m, ha, a_mask, ch, cm = setup()
    idx = torch.tensor([[0, 2, 1, 5], [3, 2, 4, 0]])  # option 2 is shorter
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    p = torch.rand(2, 4)
    for mode in ("cls", "mean"):
        m.option_vector = mode
        a, o_all = m._mix(ha, m.mix_a), m._mix(ch, m.mix_b)
        to = int(cm[idx].sum(-1).max())
        tok = cm[idx][:, :, :to] & mask[..., None]
        q, n = idx.shape
        x = m.head(o_all[idx][:, :, :to].reshape(q, n * to, -1), tok.reshape(q, n * to), a, a_mask).view(q, n, to, -1)
        vec = x[:, :, 0] if mode == "cls" else _masked_mean(x, tok)
        ref = (m.scorer(vec).squeeze(-1) + m.dual_scale * p).masked_fill(~mask, float("-inf"))
        assert torch.allclose(m(ha, a_mask, ch, cm, idx, mask, prior=p), ref, atol=1e-5), mode
