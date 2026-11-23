"""LIDM-DJ (Encoder.encode_x with option_attn='dj') against Hugging Face's own ModernBERT layers, fp32."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
NAME = "Alibaba-NLP/gte-modernbert-base"  # ModernBERT architecture, 22 layers
JOINT = 7
STATE = ("Invoice INV-2207 from Nordwind GmbH totals 4,200 EUR. The purchase order PO-118 approved 4,000 EUR for the same "
         "48 units. Delivery note DN-77 confirms 48 units received in good condition on 3 March.")
Q = "What should accounts payable do with this invoice?"
OPTS = ["approve: the invoice matches the order", "hold: ask the supplier for a corrected invoice", "reject"]
STATES, QS, OPT_SETS = [STATE, "Short state."], [Q, "Is it short?"], [OPTS, ["no", "yes"]]


@pytest.fixture(scope="module")
def enc():
    from cls.model import Encoder

    e = Encoder(NAME, option_attn="dj", joint_layers=JOINT)
    e.model.float()
    return e


def _close(x, y):
    return torch.allclose(x, y, atol=2e-3, rtol=2e-3), (x - y).abs().max().item()


def _hf(enc, ids, mask=None):
    return enc.model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states


def test_pass_a_matches_hf(enc):
    ha, a_mask, _, _, _ = enc.encode_x(STATES, QS, OPT_SETS)
    ref, ref_mask = enc.encode_a(STATES, QS)
    assert torch.equal(a_mask, ref_mask)
    for x, y in zip(ha, ref):
        ok, err = _close(x[a_mask], y[a_mask])
        assert ok, err


def test_options_match_hf_split_then_joint(enc):
    """Lower layers: the option alone (HF forward). Top JOINT layers: HF's own layer modules on [A ; option], full
    two-way attention, option positions continuing after A, local window on those positions."""
    base = enc.model
    split = len(base.layers) - JOINT
    _, _, ho, _, owner = enc.encode_x(STATES, QS, OPT_SETS)
    pairs = [(STATE, Q, t) for t in OPTS] + [("Short state.", "Is it short?", t) for t in ("no", "yes")]
    for u, (state, q, text) in enumerate(pairs):
        a_ids = enc.tok([state], [q], return_tensors="pt")["input_ids"].cuda()
        o_ids = enc.tok([f"{enc.tok.mask_token} {text}"], return_tensors="pt")["input_ids"].cuda()
        s, n = a_ids.shape[1], o_ids.shape[1]
        hs_a, hs_o = _hf(enc, a_ids), _hf(enc, o_ids)  # hidden_states[i] = input of layer i (0 = embeddings)
        for layer in range(split):
            ok, err = _close(ho[layer][u, :n], hs_o[layer + 1][0])
            assert ok, ("low", u, layer, err)
        h = torch.cat([hs_a[split], hs_o[split]], 1)
        pos = torch.arange(s + n, device="cuda")[None]
        near = ((pos[0][:, None] - pos[0][None]).abs() <= base.config.sliding_window)[None, None]
        full = torch.ones(1, 1, s + n, s + n, dtype=torch.bool, device="cuda")
        for i in range(split, len(base.layers)):
            lt = base.layers[i].attention_type
            h = base.layers[i](h, attention_mask=full if lt == "full_attention" else near,
                               position_embeddings=base.rotary_emb(h, pos, lt))
            ref = base.final_norm(h) if i == len(base.layers) - 1 else h
            ok, err = _close(ho[i][u, :n], ref[0, s:])
            assert ok, ("joint", u, i, err)
    assert owner.tolist() == [0, 0, 0, 1, 1]


def test_option_order_does_not_matter(enc):
    _, _, ho, _, _ = enc.encode_x([STATE], [Q], [OPTS])
    _, _, ho2, _, _ = enc.encode_x([STATE], [Q], [OPTS[::-1]])
    n = [len(enc.tok(f"{enc.tok.mask_token} {t}")["input_ids"]) for t in OPTS]
    for u, k in enumerate(n):
        ok, err = _close(ho[-1][u, :k], ho2[-1][len(OPTS) - 1 - u, :k])
        assert ok, err


def test_dj_batch_through_head_and_grads():
    from cls.model import LIDM, Encoder, decision_loss
    from cls.train import encode

    e = Encoder(NAME, lora_rank=4, option_attn="dj", joint_layers=JOINT)
    e.train(True)
    m = LIDM(e.d, e.heads, e.inter, e.n_layers, 1, 1, 4).cuda()
    items = [{"type": "choice", "state": STATE, "instructions": Q, "criteria": OPTS, "gold": 1},
             {"type": "noul", "state": "Short state.", "instructions": "Is it short?", "gold": 1}]
    inputs, gold, is_score, cos = encode(e, items)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = decision_loss(m(*inputs, prior=cos), gold, inputs[5], is_score)
    loss.backward()
    named = [(n, p.grad) for n, p in e.model.named_parameters() if p.requires_grad]
    assert torch.isfinite(loss) and all(g is not None and torch.isfinite(g).all() for _, g in named)
    assert all(g.abs().sum() > 0 for n, g in named if "lora_B" in n)
