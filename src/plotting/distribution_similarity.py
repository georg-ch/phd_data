from __future__ import annotations

import numpy as np
import pandas as pd


def _finite_values(values) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def wasserstein_distance_1d(x, y) -> float:
    """Return the 1D Wasserstein distance between two empirical samples."""
    x = np.sort(_finite_values(x))
    y = np.sort(_finite_values(y))
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    support = np.unique(np.concatenate([x, y]))
    if len(support) <= 1:
        return 0.0
    cdf_x = np.searchsorted(x, support, side="right") / len(x)
    cdf_y = np.searchsorted(y, support, side="right") / len(y)
    return float(np.sum(np.abs(cdf_x[:-1] - cdf_y[:-1]) * np.diff(support)))


def compare_distribution_similarity(
    x,
    y,
    *,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> dict:
    """Compute Wasserstein distance and a permutation p-value for two samples."""
    x = _finite_values(x)
    y = _finite_values(y)
    if len(x) == 0 or len(y) == 0:
        return {
            "wasserstein": float("nan"),
            "p_value": float("nan"),
            "n_x": int(len(x)),
            "n_y": int(len(y)),
            "n_permutations": int(n_permutations),
        }

    observed = wasserstein_distance_1d(x, y)
    combined = np.concatenate([x, y])
    n_x = len(x)
    rng = np.random.default_rng(random_state)

    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(len(combined))
        x_perm = combined[perm[:n_x]]
        y_perm = combined[perm[n_x:]]
        if wasserstein_distance_1d(x_perm, y_perm) >= observed - 1e-12:
            count += 1

    p_value = (count + 1) / (n_permutations + 1)
    return {
        "wasserstein": float(observed),
        "p_value": float(p_value),
        "n_x": int(len(x)),
        "n_y": int(len(y)),
        "n_permutations": int(n_permutations),
    }

