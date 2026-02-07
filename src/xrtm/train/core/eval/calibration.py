# coding=utf-8
# Copyright 2026 XRTM Team. All rights reserved.

import pickle
from pathlib import Path
from typing import List, Union

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.linear_model import LogisticRegression


class PlattScaler(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    a: float = 1.0
    b: float = 0.0
    fitted: bool = False

    def fit(self, y_true: List[int], y_prob: List[float]) -> "PlattScaler":
        eps = 1e-15
        p = np.clip(y_prob, eps, 1 - eps)
        logits = np.log(p / (1 - p)).reshape(-1, 1)
        clf = LogisticRegression(C=1e10, solver="lbfgs")
        clf.fit(logits, y_true)
        self.a = float(clf.coef_[0][0])
        self.b = float(clf.intercept_[0])
        self.fitted = True
        return self

    def transform(self, y_prob: Union[float, List[float]]) -> Union[float, List[float]]:
        if not self.fitted:
            return y_prob
        is_scalar = isinstance(y_prob, (float, int))
        y_prob_arr = np.array([y_prob]) if is_scalar else np.asarray(y_prob)
        eps = 1e-15
        p = np.clip(y_prob_arr, eps, 1 - eps)
        logits = np.log(p / (1 - p))
        scaled_logits = self.a * logits + self.b
        p_calib = 1.0 / (1.0 + np.exp(-scaled_logits))
        return float(p_calib[0]) if is_scalar else p_calib.tolist()

    def save(self, path: Union[str, Path]) -> None:
        with open(path, "wb") as f:
            pickle.dump({"a": self.a, "b": self.b, "fitted": self.fitted}, f)

    def load(self, path: Union[str, Path]) -> "PlattScaler":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.a, self.b, self.fitted = data["a"], data["b"], data["fitted"]
        return self


class BetaScaler(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    a: float = 1.0
    b: float = 1.0
    c: float = 0.0
    fitted: bool = False

    def fit(self, y_true: List[int], y_prob: List[float]) -> "BetaScaler":
        eps = 1e-15
        p = np.clip(y_prob, eps, 1 - eps)
        X = np.stack([np.log(p), -np.log(1 - p)], axis=1)
        clf = LogisticRegression(C=1e10, solver="lbfgs")
        clf.fit(X, y_true)
        self.a, self.b, self.c = float(clf.coef_[0][0]), float(clf.coef_[0][1]), float(clf.intercept_[0])
        self.fitted = True
        return self

    def transform(self, y_prob: Union[float, List[float]]) -> Union[float, List[float]]:
        if not self.fitted:
            return y_prob
        is_scalar = isinstance(y_prob, (float, int))
        y_prob_arr = np.array([y_prob]) if is_scalar else np.asarray(y_prob)
        eps = 1e-15
        p = np.clip(y_prob_arr, eps, 1 - eps)
        scaled_logits = self.a * np.log(p) - self.b * np.log(1 - p) + self.c
        p_calib = 1.0 / (1.0 + np.exp(-scaled_logits))
        return float(p_calib[0]) if is_scalar else p_calib.tolist()


__all__ = ["PlattScaler", "BetaScaler"]
