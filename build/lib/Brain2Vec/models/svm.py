# Copyright 2026 CBICA
#
# Author: Spiros Maggioros
#
# References:
#   scikit-learn: https://scikit-learn.org/stable/modules/svm.html
#
# Licenced under CBICA's non-Commercial Software Agreement:
# https://www.med.upenn.edu/cbica/software-agreement-non-commercial.html

import pickle
from typing import Any, List

from sklearn.svm import SVC, SVR


class SVMModel:

    def __init__(
        self,
        mode: str,
        C: float = 0.1,
        kernel: str = "rbf",
        max_iter: int = 10000,
        verbose: bool = False,
    ) -> None:
        """
        Initialize SVM model with either LinearSVR or SVC(depends on the kernel)

        :param C: Regularization parameter
        :type C: float(default=1.0)
        :param kernel: Either linear, rbf, poly or sigmoid
        :type kernel: str(default="rbf")
        :param max_iter: Max number of iterations for fitting
        :type max_iter: int(default=10000)
        """
        self.C = C
        self.kernel = kernel
        self.max_iter = max_iter

        if self.kernel == "linear":
            if mode == "multiclass":
                self.model = SVC(
                    max_iter=max_iter,
                    kernel="linear",
                    gamma="auto",
                    C=self.C,
                    decision_function_shape="ovr",
                    probability=True,
                    verbose=verbose,
                )
            elif mode == "binary":
                self.model = SVC(
                    max_iter=max_iter,
                    kernel="linear",
                    gamma="auto",
                    C=self.C,
                    probability=True,
                    verbose=verbose,
                )
            else:
                self.model = SVR(
                    max_iter=max_iter,
                    kernel="linear",
                    gamma="auto",
                    C=self.C,
                    verbose=verbose,
                )
        else:
            if mode == "multiclass":
                self.model = SVC(
                    max_iter=max_iter,
                    kernel=self.kernel,
                    gamma="auto",
                    C=self.C,
                    decision_function_shape="ovr",
                    probability=True,
                    verbose=verbose,
                )
            elif mode == "binary":
                self.model = SVC(
                    max_iter=max_iter,
                    kernel=self.kernel,
                    gamma="auto",
                    C=self.C,
                    probability=True,
                    verbose=verbose,
                )
            else:
                self.model = SVR(
                    max_iter=max_iter,
                    kernel=self.kernel,
                    gamma="auto",
                    C=self.C,
                    verbose=verbose,
                )

    def fit(self, X: list, y: list) -> None:
        """
        Fit the SVM model on training data

        :param X: training features
        :type X: list
        :param y: training labels
        :type y: list
        """
        self.model.fit(X, y)

    def predict(self, X: list) -> List[Any]:
        """
        Perform inference on passed list

        :param X: The passed features to perform inference
        :type X: list

        :return: The predictions
        """
        return self.model.predict(X)  # type: ignore

    def predict_proba(self, X: list) -> List[Any]:
        """
        Perform inference on passed list, but return probabilities

        :param X: The passed features to perform inference
        :type X: list

        :return: Prediction probabilities
        """
        return self.model.predict_proba(X)  # type: ignore

    def get_model(self) -> Any:
        """
        Returns the SVC model
        """
        return self.model

    def save(self, filename: str) -> None:
        """
        Saves trained model in .pkl format

        :param filename: The filename(.pkl suffix) of pre-trained model
        :type filename: str
        """
        with open(filename, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, filename: str) -> None:
        """
        Loads a pre-trained model from .pkl format

        :param filename: The filename(.pkl suffix) of pre-trained model
        :type filename: str
        """
        with open(filename, "rb") as f:
            self.model = pickle.load(f)
