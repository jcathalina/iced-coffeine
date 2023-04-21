import numpy as np
import polars as pl

from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from typing import (Optional, Any)


def _check_data(X: pl.DataFrame | np.ndarray) -> np.ndarray:
    """Helper function that checks if the data is in the right format.
    Otherwise, it ensures the data is in the right format and returns a numpy array.
    
    Parameters
    ----------
    X : pl.DataFrame | np.ndarray
        The data to check.

    Returns
    -------
    np.ndarray
        The data in the right format.

    Raises
    ------
    ValueError
    """
    out = None
    if isinstance(X, pl.DataFrame):
        X = X.to_numpy()
    
    if len(X.shape) == 3:
        out = X
    elif X.dtype == "object":
        if X.shape[1] == 1:
            values = X[:, 0]
        out = np.stack(values)
        if out.ndim == 2 and out.shape[0] == out.shape[1]:
            out = out[np.newaxis, :, :]
    if out is None:
        raise ValueError("The data is not in the right format.")
    return out


class Riemann(BaseEstimator, TransformerMixin):
    """Wrapper for the Riemannian geometry methods from pyriemann.
    
    Parameters
    ----------
    metric : str, optional
        The metric to use, by default "riemann"
    return_df : bool, optional
        Whether to return a DataFrame or a numpy array, returns a DataFrame by default.
    
    Attributes
    ----------
    tangent_space : TangentSpace
        The tangent space object from pyriemann.
    """
    def __init__(self, metric: str = "riemann", return_df: bool = True):
        self.metric = metric
        self.return_df = return_df

    def fit(self, X: pl.DataFrame | np.ndarray, y: Optional[Any] = None):
        X = _check_data(X)
        self.tangent_space = TangentSpace(metric=self.metric)
        self.tangent_space.fit(X)
        return self
    
    def transform(self, X: pl.DataFrame | np.ndarray) -> pl.DataFrame | np.ndarray:
        X = _check_data(X)
        X_tangent = self.tangent_space.transform(X)
        if self.return_df:
            X_tangent = pl.DataFrame(X_tangent)
        return X_tangent
