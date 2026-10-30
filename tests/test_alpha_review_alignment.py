"""Walk-forward joins preserve positional date/symbol identity."""

import numpy as np
import pandas as pd
import pytest

from quantcortex.alpha.factors.ml.gbdt_factor import GBDTFactor


@pytest.mark.parametrize(
    "feature_names,target_names,target_format",
    [
        (["when", "symbol"], ["date", "symbol"], "wide"),
        (["date", "ticker"], ["date", "symbol"], "wide"),
        ([None, None], ["date", "symbol"], "wide"),
        (["when", "ticker"], [None, None], "series"),
        ([None, None], ["date", "symbol"], "frame"),
        (["date", "symbol"], ["when", "symbol"], "series"),
        (["date", "symbol"], ["date", "ticker"], "frame"),
        (["date", "symbol"], ["date", "symbol"], "wide"),
    ],
)
def test_walk_forward_preserves_date_symbol_pairs(
    monkeypatch, feature_names, target_names, target_format
):
    dates = pd.date_range("2024-01-01", periods=4)
    index = pd.MultiIndex.from_product([dates, ["A", "B"]], names=feature_names)
    features = pd.DataFrame({"x": np.arange(8, dtype=float)}, index=index)
    wide = pd.DataFrame(
        [[10., 11.], [20., 21.], [30., 31.], [40., 41.]],
        index=dates, columns=["A", "B"],
    )
    if target_format == "wide":
        target = wide[["B", "A"]]
    else:
        target = wide.stack().rename_axis(target_names).iloc[::-1]
        if target_format == "frame":
            target = target.to_frame("return")
    original_features = features.copy(deep=True)
    original_target = target.copy(deep=True)
    folds = []
    model = GBDTFactor(model="sklearn")

    def record_fit(X, y):
        folds.append((X.copy(), y.copy()))
        return model

    monkeypatch.setattr(model, "fit", record_fit)
    monkeypatch.setattr(model, "predict", lambda X: X.x.to_numpy())
    result = model.fit_predict_cross_sectional(
        features, target, train_window=1, min_train_obs=2, rank_target=False
    )

    assert len(folds) == 3
    for offset, (X, y) in enumerate(folds):
        assert X.index.tolist() == [(dates[offset], "A"), (dates[offset], "B")]
        np.testing.assert_array_equal(X.x, [2 * offset, 2 * offset + 1])
        np.testing.assert_array_equal(y, [10 * (offset + 1), 10 * (offset + 1) + 1])
        np.testing.assert_array_equal(result.loc[dates[offset + 1]], [2 * (offset + 1), 2 * (offset + 1) + 1])
    pd.testing.assert_frame_equal(features, original_features)
    if isinstance(target, pd.Series):
        pd.testing.assert_series_equal(target, original_target)
    else:
        pd.testing.assert_frame_equal(target, original_target)


def test_walk_forward_future_labels_cannot_change_earlier_predictions():
    dates = pd.date_range("2024-01-01", periods=4)
    features = pd.DataFrame(
        {"x": [0., 1., 2.] * 4},
        index=pd.MultiIndex.from_product([dates, list("ABC")], names=["when", "symbol"]),
    )
    labels = pd.DataFrame([[-.01, 0., .01]] * 4, index=dates, columns=list("ABC"))
    changed = labels.copy()
    changed.iloc[-1] = [.1, .2, .3]

    def predict(target, frame=features):
        return GBDTFactor(
            model="sklearn", max_iter=5, min_samples_leaf=1,
            l2_regularization=0, early_stopping=False,
        ).fit_predict_cross_sectional(frame, target, train_window=1, min_train_obs=3)

    baseline = predict(labels)
    perturbed = predict(changed)
    canonical = predict(labels, features.rename_axis(["date", "symbol"]))
    np.testing.assert_array_equal(baseline.iloc[1], perturbed.iloc[1])
    pd.testing.assert_frame_equal(baseline, canonical)
    assert np.isfinite(baseline.iloc[1]).all()


@pytest.mark.parametrize("side", ["features", "series", "frame"])
def test_walk_forward_rejects_extra_index_levels(side):
    dates = pd.date_range("2024-01-01", periods=3)
    normal = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "symbol"])
    extra = pd.MultiIndex.from_product([dates, ["A", "B"], ["revision"]])
    features = pd.DataFrame({"x": np.arange(6)}, index=extra if side == "features" else normal)
    target = pd.Series(np.arange(6), index=normal if side == "features" else extra)
    if side == "frame":
        target = target.to_frame("return")
    with pytest.raises(ValueError, match="two.*levels"):
        GBDTFactor(model="sklearn").fit_predict_cross_sectional(
            features, target, train_window=1, min_train_obs=2
        )
