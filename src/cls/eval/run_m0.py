"""Run M0 baselines through the shared harness:  python -m cls.eval.run_m0 --model {laya,laya-b3,laya-tourn,laya-short-bge,laya-b4,laya-td,biencoder-mb,biencoder-bge}"""
from __future__ import annotations

import argparse

import numpy as np

from cls.eval import harness
from cls.tracking import ROOT


class BiEncoder:
    """Cosine bi-encoder: softmax(cos(state, option)/tau) over options. No decision head."""

    def __init__(self, name: str, pool: str, tau: float | None = None, prefix_q: str = ""):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name, attn_implementation="sdpa", dtype=torch.bfloat16).cuda().eval()
        self.pool, self.tau, self.prefix_q = pool, tau, prefix_q
        self.cache: dict[str, np.ndarray] = {}
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def embed(self, texts, max_length=512):
        torch = self.torch
        enc = self.tok(list(texts), padding=True, truncation=True, max_length=max_length, return_tensors="pt").to("cuda")
        with torch.no_grad():
            h = self.model(**enc).last_hidden_state.float()
        if self.pool == "cls":
            v = h[:, 0]
        else:
            m = enc["attention_mask"].unsqueeze(-1).float()
            v = (h * m).sum(1) / m.sum(1)
        return torch.nn.functional.normalize(v, dim=-1).cpu().numpy()

    def option_vecs(self, item):
        names = list(item["criteria"]) if item["type"] != "noul" else ["no", "yes"]
        if item["type"] == "score":
            names = [f"sentiment: {n}" for n in names]
        miss = [n for n in names if n not in self.cache]
        if miss:
            for n, v in zip(miss, self.embed(miss)):
                self.cache[n] = v
        return np.stack([self.cache[n] for n in names])

    def scores(self, item):
        q = self.embed([self.prefix_q + item["state"]])[0]
        return self.option_vecs(item) @ q

    def fit_tau(self, items):
        S = [self.scores(it) for it in items]
        best = None
        for tau in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3):
            nll = 0.0
            for s, it in zip(S, items):
                z = s / tau
                z = z - z.max()
                nll -= z[it["gold"]] - np.log(np.exp(z).sum())
            if best is None or nll < best[0]:
                best = (nll, tau)
        self.tau = best[1]
        return best[1]

    def predictor(self):
        def fn(item):
            z = self.scores(item) / self.tau
            z = np.exp(z - z.max())
            return z / z.sum()
        return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--no-track", action="store_true")
    a = ap.parse_args()

    import laya
    import torch

    from cls.data import typed

    tasks = harness.default_tasks(a.quick)
    m = a.model
    tags = {"model": m, "milestone": "M0", "stage": "baseline"}
    params = {"model": m, "instruction_style": "humanized label names, no descriptions"}
    n_params = None
    if m in ("laya", "laya-b3", "laya-tourn", "laya-short-bge", "laya-td", "laya-b4"):
        ckpt = {"laya-td": "convaiinnovations/laya-typed-decisions", "laya-b4": str(ROOT / "checkpoints" / "laya-b4")}.get(m, "convaiinnovations/laya")
        agent = laya.load(ckpt)
        n_params = sum(p.numel() for p in agent.model.parameters())
        params.update({"checkpoint": ckpt, "cfg_max_len": agent.cfg.get("max_len"), "cfg_head_max_len": agent.cfg.get("head_max_len")})
        if m == "laya-b3":
            over = {"max_len": 1024, "head_max_len": 512}
            params.update({f"override_{k}": v for k, v in over.items()})
            pred = harness.laya_predictor(agent, **over)
            tags["baseline_id"] = "B3"
        elif m == "laya-tourn":
            pred = harness.laya_tournament_predictor(agent, 16)
            params["tournament_group"] = 16
            tags["baseline_id"] = "B2"
        elif m == "laya-short-bge":
            be = BiEncoder("BAAI/bge-base-en-v1.5", "cls")
            cache: dict[str, np.ndarray] = {}

            def embed(texts):
                texts = list(texts)
                out = []
                for i, t in enumerate(texts):
                    key = t if i > 0 else None
                    if key is not None and key in cache:
                        out.append(cache[key])
                    else:
                        v = be.embed([t])[0]
                        if key is not None:
                            cache[key] = v
                        out.append(v)
                return np.stack(out)

            pred = harness.laya_shortlist_predictor(agent, embed, 20)
            params.update({"shortlist_k": 20, "embedder": "BAAI/bge-base-en-v1.5"})
            tags["baseline_id"] = "B1b"
        else:
            pred = harness.laya_predictor(agent)
            tags["baseline_id"] = {"laya": "B0", "laya-td": "B6", "laya-b4": "B4"}[m]
    elif m.startswith("biencoder"):
        name, pool, pref = {"biencoder-mb": ("answerdotai/ModernBERT-large", "mean", ""),
                            "biencoder-bge": ("BAAI/bge-base-en-v1.5", "cls", "")}[m]
        be = BiEncoder(name, pool, prefix_q=pref)
        calib = [it for it in typed.build_train(seed=5, scale=0.1) if it["task"] in ("clinc", "ag_news", "emotion")][:300]
        tau = be.fit_tau(calib)
        params.update({"encoder": name, "pool": pool, "tau_fit_on": "300 items of the training mixture", "tau": tau})
        pred, n_params = be.predictor(), be.n_params
        tags["baseline_id"] = "B5" if m == "biencoder-mb" else "B5b"
    else:
        raise SystemExit(f"unknown model {m}")
    flat = harness.run_eval(f"m0-{m}" + ("-quick" if a.quick else ""), pred, params, tags, tasks,
                            track=not a.no_track, n_params=n_params)
    print({k: round(v, 4) for k, v in flat.items() if k.endswith("/accuracy") or k.startswith("latency")})


if __name__ == "__main__":
    main()
