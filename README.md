# LIDM — late-interaction typed-decision model

A small encoder-based zero-shot decision model in the style of TypeSafe's **Jev** and Convai's **Laya**:
given a state (text/JSON), a question, and N candidate options with natural-language descriptions, return
calibrated probabilities over the options in a single non-autoregressive forward pass. No text generation.

This project tests a **late-interaction + looped-head** variant of Laya that targets Laya's known
weaknesses (high option counts, fixed option token budget) while keeping inference latency in Laya's range
(~tens of ms per question on a single GPU).

- Goals and success criteria: [docs/GOALS.md](docs/GOALS.md)
- Architecture and design hypotheses: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Experiment plan, metrics, VRAM budget: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
- Original handoff notes (verified vs. unverified claims): [docs/HANDOFF.md](docs/HANDOFF.md)

## Environment

Python 3.13, managed with `uv`. Dev machine: RTX 4070 SUPER (12 GB).

```bash
uv sync                 # create/refresh .venv
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run pytest
```

Key libs: torch, transformers, peft (LoRA), bitsandbytes (8-bit Adam), datasets, accelerate,
scikit-learn (metrics), laya (baseline), jupyter.

Verified configs: ModernBERT-large d=1024, 28 layers; mmBERT-base d=768, 22 layers.

## Status

M0 complete: eight baselines (Laya variants, fine-tuned Laya B4, bi-encoders) logged to MLflow — see [docs/RESULTS.md](docs/RESULTS.md). Next: M1 (LIDM model code).

```bash
uv run python scripts/prepare_data.py                  # build training mixture (Banking77 held out)
uv run python -m cls.eval.laya_finetune                # B4: fine-tune Laya (~25 min)
scripts/run_m0_all.sh                                  # all baselines through the shared harness
uv run python scripts/report.py                       # regenerate docs/baseline_table.md from MLflow
scripts/mlflow_ui.sh                                   # browse all experiments at http://localhost:5000
```
