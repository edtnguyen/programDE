"""
Propensity score models for guide-union indicators.
"""

from typing import Any, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_propensity_logistic(
    C: np.ndarray,
    y01: np.ndarray,
    penalty: str = "l2",
    C_value: float = 1.0,
    max_iter: int = 200,
    n_jobs: int = 1,
    **kwargs: Any,
) -> Tuple[np.ndarray, LogisticRegression]:
    """
    Fit logistic regression P(union=1 | C). Returns probabilities and model.
    """
    y = np.asarray(y01, dtype=np.int8)
    clf = LogisticRegression(
        penalty=penalty,
        C=C_value,
        solver="lbfgs",
        max_iter=max_iter,
        n_jobs=n_jobs,
        **kwargs,
    )
    clf.fit(C, y)
    p = clf.predict_proba(C)[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return p, clf
