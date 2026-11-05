"""Regenerate docs/baseline_table.md from MLflow (experiment 'baselines', runs named m0-*)."""
import mlflow

from cls.tracking import DEFAULT_URI, ROOT

mlflow.set_tracking_uri(DEFAULT_URI)
df = mlflow.search_runs(experiment_names=["baselines"], filter_string="attributes.status = 'FINISHED'")
df = df[df["tags.mlflow.runName"].str.startswith("m0-") & ~df["tags.mlflow.runName"].str.endswith("-quick")]
df = df.sort_values("start_time").drop_duplicates("tags.mlflow.runName", keep="last")
order = ["B0", "B3", "B1b", "B2", "B4", "B5", "B5b", "B6"]
df["_o"] = df["tags.baseline_id"].map(lambda x: order.index(x) if x in order else 99)
df = df.sort_values("_o")

acc_cols = [("banking77_N77", "B77 N=77"), ("banking77_N40", "N=40"), ("banking77_N20", "N=20"),
            ("banking77_N10", "N=10"), ("banking77_N5", "N=5"), ("boolq_val", "BoolQ"), ("sst5_test", "SST-5")]


def f(v, d=3):
    return "-" if v != v else f"{v:.{d}f}"


lines = ["### Accuracy", "",
         "| id | run | " + " | ".join(n for _, n in acc_cols) + " | p50 ms | VRAM GB | mlflow run |",
         "|" + "---|" * (len(acc_cols) + 5)]
for _, r in df.iterrows():
    cells = [f(r[f"metrics.{t}/accuracy"]) for t, _ in acc_cols]
    lines.append(f"| {r['tags.baseline_id']} | {r['tags.mlflow.runName']} | " + " | ".join(cells) +
                 f" | {f(r['metrics.latency_ms_p50'], 1)} | {f(r['metrics.peak_vram_gb'], 2)} | {r['run_id'][:8]} |")
lines += ["", "### Calibration (ECE, 15 bins) and NLL", "",
          "| id | " + " | ".join(f"{n} ECE / NLL" for _, n in acc_cols) + " |", "|" + "---|" * (len(acc_cols) + 1)]
for _, r in df.iterrows():
    cells = [f"{f(r[f'metrics.{t}/ece'], 2)} / {f(r[f'metrics.{t}/nll'], 2)}" for t, _ in acc_cols]
    lines.append(f"| {r['tags.baseline_id']} | " + " | ".join(cells) + " |")
(ROOT / "docs" / "baseline_table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
