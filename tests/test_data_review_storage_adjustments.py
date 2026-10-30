"""Regression checks for partition identity and adjustment input validation."""

import numpy as np
import pandas as pd
import pytest

from quantcortex.data.processors.adjustments import AdjustmentError, AdjustmentValidator
from quantcortex.data.storage.parquet_store import ParquetStore


def test_partitioned_parquet_preserves_symbol_identity_and_string_filters(tmp_path):
    frame = pd.DataFrame(
        {"symbol": ["000001", "1", "000002"], "close": [10.0, 20.0, 30.0]}
    )
    store = ParquetStore(tmp_path)
    store.write(frame, "prices", partition_cols=["symbol"])

    result = store.read("prices").sort_values("close").reset_index(drop=True)
    assert result.symbol.tolist() == ["000001", "1", "000002"]
    pd.testing.assert_frame_equal(result.loc[:, frame.columns], frame)
    selected = store.read(
        "prices", columns=["symbol", "close"], filters=[("symbol", "==", "000001")]
    )
    pd.testing.assert_frame_equal(selected, frame.iloc[:1])


@pytest.mark.parametrize(
    "partition",
    [
        pd.Series([1, 2], dtype="int64"),
        pd.Series(pd.to_datetime(["2024-01-02", "2024-01-03"])),
        pd.Series(pd.Categorical(["000001", "000002"])),
    ],
    ids=["integer", "timestamp", "categorical-symbol"],
)
def test_partitioned_parquet_retains_partition_types_after_append(tmp_path, partition):
    frame = pd.DataFrame({"partition": partition, "value": [10.0, 20.0]})
    store = ParquetStore(tmp_path)
    store.write(frame.iloc[:1], "data", partition_cols=["partition"])
    store.write(frame.iloc[1:], "data", partition_cols=["partition"], mode="append")

    result = store.read("data").sort_values("value").reset_index(drop=True)

    pd.testing.assert_frame_equal(result.loc[:, frame.columns], frame)


def test_parquet_read_retains_legacy_unmarked_dataset_support(tmp_path):
    path = tmp_path / "legacy"
    path.mkdir()
    frame = pd.DataFrame({"close": [100.0]})
    frame.to_parquet(path / "part.parquet", index=False)

    pd.testing.assert_frame_equal(ParquetStore(tmp_path).read("legacy"), frame)


@pytest.mark.parametrize("side", ["raw", "adjusted"])
@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan, 0.0, -1.0])
def test_adjustment_validator_rejects_nonfinite_or_nonpositive_inputs(side, bad):
    index = pd.date_range("2024-01-01", periods=2)
    raw = pd.Series([100.0, 100.0], index=index)
    adjusted = raw.copy()
    if side == "raw":
        raw[:] = bad
    else:
        adjusted[:] = bad

    with pytest.raises(AdjustmentError, match="finite and positive"):
        AdjustmentValidator().validate_adjustment(raw, adjusted)


def test_adjustment_validator_preserves_split_dividend_and_normalization_rules():
    index = pd.date_range("2024-01-01", periods=4)
    raw = pd.Series([100.0, 50.0, 49.0, 50.0], index=index)
    # 2:1 split on day 2; $1 dividend on day 3 against prior close of $50.
    splits = pd.Series([2.0], index=index[1:2])
    dividends = pd.Series([1.0], index=index[2:3])
    adjusted = pd.Series([49.0, 49.0, 49.0, 50.0], index=index)
    validator = AdjustmentValidator()

    assert validator.validate_adjustment(raw, adjusted, splits, dividends)
    assert validator.validate_adjustment(raw, adjusted * 0.5, splits, dividends)
    with pytest.raises(AdjustmentError, match="inconsistent with corporate actions"):
        validator.validate_adjustment(raw, raw, splits, dividends)
