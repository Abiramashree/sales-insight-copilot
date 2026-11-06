"""Small, dependency-light high-dimensional fixed-effects OLS.

Absorbs any number of fixed effects by alternating projections (demeaning),
which is exact for unbalanced panels at convergence, then runs OLS with
cluster-robust standard errors. Enough for stacked DiD without building
thousands of dummy columns.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def demean(df: pd.DataFrame, cols: list[str], fe: list[str], tol: float = 1e-8,
           max_iter: int = 200) -> pd.DataFrame:
    out = df[cols].astype(float).copy()
    keys = [df[f] for f in fe]
    for _ in range(max_iter):
        prev = out.values.copy()
        for k in keys:
            out = out - out.groupby(k).transform("mean")
        if np.max(np.abs(out.values - prev)) < tol:
            break
    return out


@dataclass
class FEResult:
    params: pd.Series
    vcov: pd.DataFrame
    nobs: int
    nclusters: int

    @property
    def se(self) -> pd.Series:
        return pd.Series(np.sqrt(np.diag(self.vcov)), index=self.params.index)

    def ci(self, z: float = 1.96) -> pd.DataFrame:
        return pd.DataFrame({"lo": self.params - z * self.se, "hi": self.params + z * self.se})

    def wald(self, names: list[str]) -> tuple[float, int]:
        """Joint test that params[names] == 0. Returns (chi2 stat, dof)."""
        b = self.params[names].values
        V = self.vcov.loc[names, names].values
        return float(b @ np.linalg.solve(V, b)), len(names)


def fe_ols(df: pd.DataFrame, y: str, x: list[str], fe: list[str], cluster: str) -> FEResult:
    d = demean(df, [y, *x], fe)
    Y, X = d[y].values, d[x].values
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    u = Y - X @ beta
    scores = pd.DataFrame(X * u[:, None]).groupby(df[cluster].values).sum().values
    G, N, K = scores.shape[0], len(Y), len(x)
    meat = scores.T @ scores
    adj = G / (G - 1) * (N - 1) / (N - K)
    V = adj * XtX_inv @ meat @ XtX_inv
    return FEResult(pd.Series(beta, index=x), pd.DataFrame(V, index=x, columns=x), N, G)
