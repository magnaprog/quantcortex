"""Independent oracles for refit state, feature numerics and factor buckets."""

import numpy as np
import pandas as pd
import pytest

from quantcortex.alpha.factors.ml.gbdt_factor import GBDTFactor
from quantcortex.alpha.factors.ml.neural_factor import NeuralFactor
from quantcortex.alpha.feature_engineering.alpha158 import Alpha158
from quantcortex.alpha.validation.alphalens_report import (
    AlphalensReport,
    quantile_returns,
)


@pytest.mark.filterwarnings("ignore:Stochastic Optimizer:sklearn.exceptions.ConvergenceWarning")
@pytest.mark.parametrize("kind", ["gbdt_constructor", "gbdt_fit", "neural_fit"])
def test_failed_refit_preserves_complete_previous_model(monkeypatch, kind):
    X = pd.DataFrame({"a": np.tile([0., 1., 2.], 20), "b": np.tile([10., 0., 0.], 20)})
    y = pd.Series(X.a.to_numpy())
    if kind == "neural_fit":
        model = NeuralFactor(hidden=(4,), epochs=3).fit(X, y)
        estimator = model.model_
        scaler = model.scaler_
    else:
        model = GBDTFactor(
            model="sklearn", max_iter=5, min_samples_leaf=1,
            l2_regularization=0, early_stopping=False,
        ).fit(X, y)
        estimator = model.estimator_
    names = model.feature_names_.copy()
    backend = model.backend_
    before = model.predict(X)

    def fail(*args, **kwargs):
        raise RuntimeError("refit failed")

    if kind == "gbdt_constructor":
        monkeypatch.setattr(model, "_build_estimator", fail)
    elif kind == "gbdt_fit":
        class BrokenEstimator:
            fit = fail
        monkeypatch.setattr(model, "_build_estimator", lambda: ("sklearn", BrokenEstimator()))
    else:
        monkeypatch.setattr(model, "_fit_sklearn", fail)
    with pytest.raises(RuntimeError, match="refit failed"):
        model.fit(X[["b", "a"]] + 1000, y + 10)

    assert model.feature_names_ == names
    assert model.backend_ == backend
    if kind == "neural_fit":
        assert model.model_ is estimator
        assert model.scaler_ is scaler
    else:
        assert model.estimator_ is estimator
    np.testing.assert_array_equal(model.predict(X), before)


def _ohlcv(close):
    close = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * .99,
         "close": close, "volume": np.arange(len(close)) + 100.},
        index=pd.date_range("2024-01-01", periods=len(close)),
    )


@pytest.mark.parametrize("scale", [1., 1e-6, 1e6])
def test_alpha158_linear_trend_r_squared_is_scale_invariant(scale):
    features = Alpha158(windows=(5,)).compute(_ohlcv(scale * (1 + .1 * np.arange(10))))
    assert features.RSQR5.iloc[:4].isna().all()
    np.testing.assert_allclose(features.RSQR5.iloc[4:], 1., atol=1e-12)


def test_alpha158_centered_regression_handles_large_offset_and_constant_windows():
    close = 100 + 1e-6 * np.arange(10)
    slope, r_squared, residual = Alpha158._rolling_regression(pd.Series(close), 5)
    np.testing.assert_allclose(slope.iloc[4:], 1e-6, rtol=1e-8)
    np.testing.assert_allclose(r_squared.iloc[4:], 1., atol=1e-12)
    np.testing.assert_allclose(residual.iloc[4:], 0., atol=1e-12)
    constant = Alpha158(windows=(5,)).compute(_ohlcv(np.full(10, 1e-6)))
    assert constant.RSQR5.isna().all()
    np.testing.assert_array_equal(constant.BETA5.iloc[4:], 0.)
    np.testing.assert_array_equal(constant.RESI5.iloc[4:], 0.)


def test_alpha158_regression_matches_independent_least_squares_and_missing_window():
    values = pd.Series([10., 11., 10., 13., 12., np.nan, 20.])
    slope, r_squared, residual = Alpha158._rolling_regression(values, 5)
    design = np.column_stack([np.ones(5), np.arange(5)])
    coef = np.linalg.lstsq(design, values.iloc[:5], rcond=None)[0]
    errors = values.iloc[:5].to_numpy() - design @ coef
    expected_r2 = 1 - np.dot(errors, errors) / np.sum((values.iloc[:5] - values.iloc[:5].mean()) ** 2)
    assert slope.iloc[4] == pytest.approx(coef[1])
    assert r_squared.iloc[4] == pytest.approx(expected_r2)
    assert residual.iloc[4] == pytest.approx(errors[-1])
    assert slope.iloc[5:].isna().all()
    assert r_squared.iloc[5:].isna().all()


def test_quantiles_keep_factor_universe_when_a_return_column_is_absent():
    dates = pd.DatetimeIndex(["2024-01-01"])
    factor = pd.DataFrame([[1., 2., 3., 4., 5., 6.]], index=dates, columns=list("ABCDEF"))
    returns = pd.DataFrame([[0., np.nan, .02, .08, .09, .10]], index=dates, columns=factor.columns)
    expected = quantile_returns(factor, returns, quantiles=2)
    actual = quantile_returns(factor, returns.drop(columns="B"), quantiles=2)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.loc[1, "mean_return"] == pytest.approx(.01)
    assert actual.loc[2, "mean_return"] == pytest.approx(.09)
    assert actual.loc["long_short", "mean_return"] == pytest.approx(.08)
    report = AlphalensReport(factor, returns.drop(columns="B"), quantiles=2).compute()
    assert report["long_short_return"] == pytest.approx(.08)
