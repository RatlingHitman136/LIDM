# LIDM — late-interaction typed-decision model

A small encoder-based zero-shot decision model in the style of TypeSafe's **Jev** and Convai's **Laya**:
given a state (text/JSON), a question, and N candidate options with natural-language descriptions, return
calibrated probabilities over the options in a single non-autoregressive forward pass. No text generation.

This project tests a **late-interaction + looped-head** variant of Laya that targets Laya's known
weaknesses (high option counts, fixed option token budget) while keeping inference latency in Laya's range
(~tens of ms per question on a single GPU).

- Goals and success criteria: [docs/GOALS.md](docs/GOALS.md)
- Initial design and its evidence (snapshot of the design doc): [docs/DESIGN.md](docs/DESIGN.md)
- Dataset survey behind the training plan (snapshot): [docs/DATASETS.md](docs/DATASETS.md)
- What is built, and where it departs from the design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- Experiment plan, stages, gates, VRAM: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)
- Results (M0 baselines, M1): [docs/RESULTS.md](docs/RESULTS.md)
- Backlog (what to try later, focus, open issues): [docs/BACKLOG.md](docs/BACKLOG.md)
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

- **M0 done:** eight baselines (Laya variants, fine-tuned Laya B4, bi-encoders) in MLflow.
- **M1 done:** runs B / C / D on the M0 mixture with a frozen encoder. After a layer-normalisation fix they
  reach ~0.75 in-distribution but stay near chance on Banking77 zero-shot (labels memorised; 152 distinct
  labels in the data). Looping gave no gain there. See [docs/RESULTS.md](docs/RESULTS.md).
- **Stage 1 done:** label reading on 974k questions. Frozen, the model memorised labels (Banking77 0.044);
  with encoder LoRA r=16 it reads unseen labels: **Banking77 0.434 @N=77, 0.641 @N=20** (shipped Laya 0.360;
  bge bi-encoder 0.695). LoRA is now used from stage 1 on. Next options: [docs/BACKLOG.md](docs/BACKLOG.md).

```bash
# M0
uv run python scripts/prepare_data.py                  # M0 training mixture (Banking77 held out)
uv run python -m cls.eval.laya_finetune                # B4: fine-tune Laya (~25 min)
scripts/run_m0_all.sh                                  # all baselines through the shared harness
uv run python scripts/report.py                        # regenerate docs/baseline_table.md from MLflow
scripts/mlflow_ui.sh                                   # browse all experiments at http://localhost:5000
# M1
uv run python -m cls.train --run B                     # runs B/C/D on the M0 mixture, MLflow experiment `m1`
# Stage 1
uv run python scripts/prepare_stage1.py                # data/stage1_{train,val}.jsonl
uv run python -m cls.train --run B --stage1 --limit 64000 --name stage1-B-pilot   # pilot / LR slice
uv run python -m cls.train --run B --stage1            # full pass, MLflow experiment `stage1`
uv run python -m cls.train --run B --stage1 --encoder-lora 16   # stage 1 with encoder LoRA (~7 h)
uv run python scripts/eval_banking77.py checkpoints/stage1-Blora/last.pt --n 3080   # Banking77 test
uv run python scripts/eval_lidm.py checkpoints/stage1-Blora/last.pt --name stage1-Blora   # full M0 task set
```
