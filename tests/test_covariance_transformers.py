import numpy as np
import polars as pl
import pytest

from iced_coffeine.covariance_transformers import Riemann

n_subjects = 10
n_channels = 4


@pytest.fixture
def toy_data(request):
    X_cov = np.random.randn(n_subjects, n_channels, n_channels)
    for sub in range(n_subjects):
        X_cov[sub] = X_cov[sub] @ X_cov[sub].T
    if request:
        X_cov = pl.DataFrame({"cov": X_cov})
    return X_cov


@pytest.mark.parametrize("toy_data", [True, False], indirect=True)
def test_Riemann_transformer_with_ndarray_output(toy_data):
    riemann = Riemann(return_df=False)
    Xt_cov = riemann.fit_transform(toy_data)
    assert Xt_cov.shape == (n_subjects, n_channels * (n_channels + 1) // 2)


@pytest.mark.parametrize("toy_data", [True, False], indirect=True)
def test_Riemann_transformer_with_dataframe_output(toy_data):
    riemann = Riemann(return_df=True)
    Xt_cov = riemann.fit_transform(toy_data)
    assert Xt_cov.shape == (n_subjects, n_channels * (n_channels + 1) // 2)
