"""Few-shot embedding-probe baselines on Banking77, as MTEB scores classification: logistic regression on frozen,
normalised sentence embeddings of the k-shot training examples, evaluated on the full test split (3,080).

    uv run python scripts/fewshot_probe.py --train data/fewshot/b77_k8_s0_train.jsonl ... \\
        --encoders BAAI/bge-base-en-v1.5 Alibaba-NLP/gte-modernbert-base
"""
import argparse
import json

import numpy as np
from sklearn.linear_model import LogisticRegression

from cls.data.banking77 import load_banking77
from cls.model import Encoder
from cls.train import read_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--train", nargs="+", required=True, help="few-shot training files (one per seed)")
ap.add_argument("--encoders", nargs="+", default=["BAAI/bge-base-en-v1.5", "Alibaba-NLP/gte-modernbert-base"])
a = ap.parse_args()

test = load_banking77("test")
for name in a.encoders:
    enc = Encoder(name)
    X_test = np.concatenate([enc.embed(test.texts[i:i + 128]) for i in range(0, len(test.texts), 128)])
    accs = []
    for path in a.train:
        items = read_jsonl(path)
        X = np.concatenate([enc.embed([it["state"] for it in items[i:i + 128]]) for i in range(0, len(items), 128)])
        clf = LogisticRegression(max_iter=100).fit(X, [it["gold"] for it in items])  # MTEB's probe settings
        accs.append(float(np.mean(clf.predict(X_test) == np.array(test.labels))))
        print(json.dumps({"encoder": name, "train": path, "test_acc_N77": round(accs[-1], 4)}), flush=True)
    print(json.dumps({"encoder": name, "mean": round(float(np.mean(accs)), 4), "std": round(float(np.std(accs)), 4)}), flush=True)
