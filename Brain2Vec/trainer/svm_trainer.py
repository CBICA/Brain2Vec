from typing import Any

import numpy as np
import optuna
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from Brain2Vec.models.svm import SVMModel


def compute_metrics(
    y_preds: list,
    y_hat: list,
    mode: str,
    y_preds_proba: list = [],
) -> dict:
    """
    Return needed metrics for classification or regression.

    :param y_preds: The model predictions
    :param y_hat: Ground truth
    :param mode: "binary", "multiclass", or "regression"
    :param y_preds_proba: Predicted probabilities — only used for classification
    :return: Dictionary of evaluation metrics
    """
    y_hat_np = np.array(y_hat)
    y_preds_np = np.array(y_preds)

    if mode == "regression":
        return {
            "MSE": mean_squared_error(y_hat_np, y_preds_np),
            "RMSE": np.sqrt(mean_squared_error(y_hat_np, y_preds_np)),
            "MAE": mean_absolute_error(y_hat_np, y_preds_np),
            "R2": r2_score(y_hat_np, y_preds_np),
        }

    y_preds_proba_np = np.array(y_preds_proba)
    auc = 0.0
    if y_preds_proba_np.size > 0:
        if mode == "binary":
            auc = roc_auc_score(y_hat_np, y_preds_proba_np[:, 1])
        else:
            auc = roc_auc_score(y_hat_np, y_preds_proba_np, multi_class="ovr")

    return {
        "F1": f1_score(y_hat_np, y_preds_np, average="macro"),
        "Accuracy": accuracy_score(y_hat_np, y_preds_np),
        "AUC": auc,
    }


def train(
    X_train: list,
    X_test: list,
    y_train: list,
    y_test: list,
    mode: str,
    save_to: str = "svm_best.pkl",
) -> Any:
    """
    SVM trainer for classification or regression.

    :param embeddings: Embeddings computed by a foundation model
    :param labels: Labels/targets for each embedding
    :param mode: "binary", "multiclass", or "regression"
    :return: Tuple of (best_model, stats)
    """
    if mode not in ("binary", "multiclass", "regression"):
        raise ValueError(
            f"mode must be 'binary', 'multiclass', or 'regression', got '{mode}'"
        )

    def objective(trial: Any) -> float:
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
        svc_c = trial.suggest_float("C", 1e-1, 100, log=True)

        if mode == "regression":
            model = SVMModel(mode=mode, C=svc_c, kernel=kernel)
            pipeline = Pipeline([("svm", model.get_model())])
            scoring = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
            scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=5)

            checkpoint = {
                "trial_params": trial.params,
                "MSE": -scores["test_neg_mean_squared_error"].mean(),
                "MAE": -scores["test_neg_mean_absolute_error"].mean(),
                "R2": scores["test_r2"].mean(),
            }
            trial.set_user_attr("checkpoint", checkpoint)
            return float(scores["test_r2"].mean())

        else:
            model = SVMModel(mode=mode, C=svc_c, kernel=kernel)
            pipeline = Pipeline([("svm", model.get_model())])
            auc_scorer = "roc_auc" if mode == "binary" else "roc_auc_ovr"
            scoring = ["f1_macro", auc_scorer, "accuracy"]
            scores = cross_validate(pipeline, X_train, y_train, scoring=scoring, cv=5)

            checkpoint = {
                "trial_params": trial.params,
                "f1_score": scores["test_f1_macro"].mean(),
                "AUC": scores[f"test_{auc_scorer}"].mean(),
                "Accuracy": scores["test_accuracy"].mean(),
            }
            trial.set_user_attr("checkpoint", checkpoint)
            return float(scores["test_f1_macro"].mean())

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, n_jobs=4)

    best_trial = study.best_trial
    best_checkpoint = best_trial.user_attrs["checkpoint"]
    best_hyperparams = best_checkpoint["trial_params"]

    if mode == "regression":
        best_model = SVMModel(
            mode=mode,
            C=best_hyperparams["C"],
            kernel=best_hyperparams["kernel"],
        )
        best_model.fit(X_train, y_train)
        y_preds = best_model.predict(X_test)
        stats = compute_metrics(y_preds=y_preds, y_hat=y_test, mode=mode)

    else:
        best_model = SVMModel(
            mode=mode,
            C=best_hyperparams["C"],
            kernel=best_hyperparams["kernel"],
        )
        best_model.fit(X_train, y_train)
        y_preds = best_model.predict(X_test)
        y_preds_proba = best_model.predict_proba(X_test)
        stats = compute_metrics(
            y_preds=y_preds, y_hat=y_test, mode=mode, y_preds_proba=y_preds_proba
        )

    best_model.save(save_to)
    return best_model, stats
