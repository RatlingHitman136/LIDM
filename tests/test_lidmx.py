"""LIDM-X (Encoder.encode_x) against Hugging Face's own ModernBERT forward, in fp32 on a small ModernBERT checkpoint."""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
NAME = "Alibaba-NLP/gte-modernbert-base"  # ModernBERT architecture, 22 layers
STATE = ("Invoice INV-2207 from Nordwind GmbH totals 4,200 EUR. The purchase order PO-118 approved 4,000 EUR for the same "
         "48 units. Delivery note DN-77 confirms 48 units received in good condition on 3 March.")
Q = "What should accounts payable do with this invoice?"
OPTS = ["approve: the invoice matches the order", "hold: ask the supplier for a corrected invoice", "reject"]


@pytest.fixture(scope="module")
def enc():
    from cls.model import Encoder

    e = Encoder(NAME, option_attn="state")
    e.model.float()
    return e


def _close(x, y):
    return torch.allclose(x, y, atol=2e-3, rtol=2e-3), (x - y).abs().max().item()


def test_pass_a_matches_hf(enc):
    states, qs = [STATE, "Short state."], [Q, "Is it short?"]
    ha, a_mask, _, _, _ = enc.encode_x(states, qs, [OPTS, ["no", "yes"]])
    ref, ref_mask = enc.encode_a(states, qs)
    assert torch.equal(a_mask, ref_mask) and len(ha) == len(ref)
    for x, y in zip(ha, ref):
        ok, err = _close(x[a_mask], y[a_mask])
        assert ok, err


def test_options_match_packed_hf_pass(enc):
    """Each option = HF's layers on one packed [A; option] sequence with the LIDM-X mask and shared positions."""
    ha, _, ho, o_mask, owner = enc.encode_x([STATE, "Short state."], [Q, "Is it short?"], [OPTS, ["no", "yes"]])
    assert owner.tolist() == [0, 0, 0, 1, 1]
    for u, (state, q, text) in enumerate([(STATE, Q, t) for t in OPTS] + [("Short state.", "Is it short?", t) for t in ("no", "yes")]):
        a_ids = enc.tok([state], [q], return_tensors="pt")["input_ids"]
        o_ids = enc.tok([f"{enc.tok.mask_token} {text}"], return_tensors="pt")["input_ids"]
        s, n = a_ids.shape[1], o_ids.shape[1]
        ids = torch.cat([a_ids, o_ids], 1).cuda()
        pos = torch.arange(s + n, device="cuda")
        allow = torch.zeros(s + n, s + n, dtype=torch.bool, device="cuda")
        allow[:s, :s] = True  # pass A sees only itself
        allow[s:] = True  # the option sees pass A and itself
        near = (pos[:, None] - pos[None]).abs() <= enc.model.config.sliding_window
        masks = {"full_attention": allow[None, None], "sliding_attention": (allow & near)[None, None]}
        ref = enc.model(input_ids=ids, attention_mask=masks, position_ids=pos[None], output_hidden_states=True).hidden_states[1:]
        b = owner[u].item()
        for layer, r in enumerate(ref):
            ok, err = _close(ho[layer][u, :n], r[0, s:])
            assert ok, (u, layer, err)
            ok, err = _close(ha[layer][b, :s], r[0, :s])
            assert ok, (u, layer, err)
        assert o_mask[u].sum().item() == n


def test_option_order_does_not_matter(enc):
    _, _, ho, _, _ = enc.encode_x([STATE], [Q], [OPTS])
    _, _, ho2, _, _ = enc.encode_x([STATE], [Q], [OPTS[::-1]])
    n = [len(enc.tok(f"{enc.tok.mask_token} {t}")["input_ids"]) for t in OPTS]
    for u, k in enumerate(n):
        ok, err = _close(ho[-1][u, :k], ho2[-1][len(OPTS) - 1 - u, :k])
        assert ok, err


def test_lidmx_batch_through_head_and_grads():
    """encode() + LIDM forward + backward with encoder LoRA: finite loss, every LoRA weight gets a gradient."""
    from cls.model import LIDM, Encoder, decision_loss
    from cls.train import encode

    e = Encoder(NAME, lora_rank=4, option_attn="state")
    e.train(True)
    m = LIDM(e.d, e.heads, e.inter, e.n_layers, 1, 1, 4).cuda()
    items = [{"type": "choice", "state": STATE, "instructions": Q, "criteria": OPTS, "gold": 1},
             {"type": "noul", "state": "Short state.", "instructions": "Is it short?", "gold": 1}]
    inputs, gold, is_score, cos = encode(e, items)
    assert inputs[4].tolist() == [[0, 1, 2], [3, 4, 0]] and inputs[5].tolist() == [[True] * 3, [True, True, False]]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = decision_loss(m(*inputs, prior=cos), gold, inputs[5], is_score)
    loss.backward()
    assert torch.isfinite(loss)
    named = [(n, p.grad) for n, p in e.model.named_parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for _, g in named)
    # lora_B starts at zero, so on the first step only the B matrices get gradient: every one of them must
    assert all(g.abs().sum() > 0 for n, g in named if "lora_B" in n)
