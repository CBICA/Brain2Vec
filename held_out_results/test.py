import json

import numpy as np
from scipy import stats

if __name__ == "__main__":

    acc, f1, auc, precision, recall = [], [], [], [], []
    for i in range(5):
        with open(f"diabetes_held_out{i}.json", "r") as f:
            data = json.load(f)

        _stats = data["stats"]
        acc.append(_stats["Accuracy"])
        f1.append(_stats["F1"])
        auc.append(_stats["AUC"])
        precision.append(_stats["Precision"])
        recall.append(_stats["Recall"])

    print(f"ACC: {np.mean(acc):.4f} {np.std(acc):.3f}")
    print(f"AUC: {np.mean(auc):.4f} {np.std(auc):.3f}")
    print(f"F1: {np.mean(f1):.4f} {np.std(f1):.3f}")
    print(f"Precision: {np.mean(precision):.4f} {np.std(precision):.3f}")
    print(f"Recall: {np.mean(recall):.4f} {np.std(recall):.3f}")

    n = len(auc)
    mean = np.mean(auc)
    std = np.std(auc)
    sem = std / np.sqrt(n)

    ci = stats.t.interval(
        confidence=0.95,
        df=n - 1,
        loc=mean,
        scale=sem,
    )

    print(f"Mean: {mean:.4f}")
    print(f"Std: {std:.4f}")
    print(f"95% CI: ({ci[0]:.4f}, {ci[1]:.4f})")
