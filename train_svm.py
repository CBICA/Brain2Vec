# Copyright 2026 CBICA
#
# Author: Spiros Maggioros
#
# Licenced under CBICA's non-Commercial Software Agreement:
# https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from Brain2Vec.trainer.svm_trainer import train as svm_train


def train_disease_classification(args: Any) -> None:
    X, y = [], []
    mrids = []

    with open(args.cn_embeddings, "r") as f:
        cn_data = json.load(f)
        X.extend(list(cn_data.values()))
        y.extend([0] * len(cn_data))
        mrids.extend(list(cn_data.keys()))

    with open(args.dx_embeddings, "r") as f:
        dx_data = json.load(f)
        X.extend(list(dx_data.values()))
        y.extend([1] * len(dx_data))
        mrids.extend(list(dx_data.keys()))

    mrids = np.array(mrids)
    X = np.array(X)
    y = np.array(y)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    train_mrids, test_mrids = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        train_mrids = list(mrids[train_idx])
        test_mrids = list(mrids[test_idx])

        _, stats = svm_train(
            X_train_outer,
            X_test_outer,
            y_train_outer,
            y_test_outer,
            mode="binary",
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            save_to=f"svc_brain2vec_embeddings_fold{fold_idx}.pkl",
        )
        print(f" Fold {fold_idx} stats: {stats}")

        curr_fold_data = {
            "train": train_mrids,
            "test": test_mrids,
            "stats": stats,
        }

        with open(f"fold{fold_idx}.json", "w") as f:
            json.dump(curr_fold_data, f, indent=4)


def train_ba_regression(args: Any) -> None:
    df = pd.read_csv(args.label_dict)

    with open(args.embeddings, "r") as f:
        X = json.load(f)

    mrids = np.array(list(X.keys()))
    X = np.array(list(X.values()))

    y = []
    for mrid in mrids:
        y.append(df.loc[df["MRID"] == mrid, "Age"].iloc[0])

    y = np.array(y)

    skf = KFold(n_splits=5, shuffle=True, random_state=42)

    train_mrids, test_mrids = [], []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train_outer, X_test_outer = X[train_idx], X[test_idx]
        y_train_outer, y_test_outer = y[train_idx], y[test_idx]

        train_mrids = list(mrids[train_idx])
        test_mrids = list(mrids[test_idx])

        _, stats = svm_train(
            X_train_outer,
            X_test_outer,
            y_train_outer,
            y_test_outer,
            mode="regression",
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            save_to=f"svr_ba_brain2vec_fold{fold_idx}.pkl",
        )
        print(f" Fold {fold_idx} stats: {stats}")

        curr_fold_data = {
            "train": train_mrids,
            "test": test_mrids,
            "stats": stats,
        }

        with open(f"ba_fold{fold_idx}.json", "w") as f:
            json.dump(curr_fold_data, f, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="svm_train",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    dx_classification = subparsers.add_parser(
        "dx_classification", help="Select disease classification mode"
    )

    dx_classification.add_argument(
        "--cn_embeddings",
        type=str,
        required=True,
        help="CN embeddings computed by a brain2vec encoder",
    )

    dx_classification.add_argument(
        "--dx_embeddings",
        type=str,
        required=True,
        help="DX embeddings computed by a brain2vec encoder",
    )

    dx_classification.add_argument(
        "--n_trials",
        type=int,
        required=False,
        default=50,
        help="Number of trials for optuna",
    )

    dx_classification.add_argument(
        "--n_jobs",
        type=int,
        required=False,
        default=4,
        help="Number of parallel workes for optuna trials",
    )

    dx_classification.set_defaults(func=train_disease_classification)

    ba_regression = subparsers.add_parser(
        "ba_regression", help="Select brain age regression mode"
    )

    ba_regression.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Embeddings computed by a brain2vec encoder",
    )

    ba_regression.add_argument(
        "--label_dict",
        type=str,
        required=True,
        help="CSV file with MRID and Age columns",
    )

    ba_regression.add_argument(
        "--n_trials",
        type=int,
        required=False,
        default=50,
        help="Number of trials for optuna",
    )

    ba_regression.add_argument(
        "--n_jobs",
        type=int,
        required=False,
        default=4,
        help="Number of parallel workes for optuna trials",
    )

    ba_regression.set_defaults(func=train_ba_regression)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
