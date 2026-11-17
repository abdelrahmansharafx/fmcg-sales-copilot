"""Forecast metrics and the model-vs-baseline evaluation table.

Run on its own to re-score the saved model on the test period:
    uv run python -m src.forecast.evaluate
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sqlalchemy as sa

from src.config import DB_URL, MODELS_DIR, TRUE_DEMAND_PATH
from src.forecast.features import FEATURES, load_panel, make_supervised, time_split


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Weighted absolute % error = sum|error| / sum(actual). Our headline metric.

    Robust to zero-sales weeks and weights big outlets more, like the business does.
    """
    return float(np.abs(actual - predicted).sum() / actual.sum() * 100)


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute % error, only over weeks with actual > 0 (it is undefined at 0).

    Zero weeks are mostly stock-outs nobody can forecast. MAPE is also dominated by
    low-volume kiosks, which is why we report WAPE alongside it.
    """
    nonzero = actual > 0
    return float((np.abs(actual - predicted)[nonzero] / actual[nonzero]).mean() * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root mean squared error, in units. Punishes big misses more than small ones."""
    return float(np.sqrt(((actual - predicted) ** 2).mean()))


def score(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """All three metrics for one set of predictions."""
    a, p = actual.to_numpy(float), np.asarray(predicted, dtype=float)
    return {"MAPE %": mape(a, p), "RMSE": rmse(a, p), "WAPE %": wape(a, p)}


ORACLE = "Oracle (true mean), unreachable floor"

# Display name -> column of the scored frame holding that method's prediction.
PREDICTION_COLUMNS = {
    "Seasonal naive": "seasonal_naive",
    "Moving avg (4w)": "mean_4",
    "LightGBM": "model",
    ORACLE: "oracle",
}


def load_true_demand(path: Path = TRUE_DEMAND_PATH) -> pd.DataFrame:
    """The simulator's true expected demand per outlet x SKU x week (written by generate_data)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: uv run python scripts/generate_data.py")
    true_demand = pd.read_csv(path, parse_dates=["week_start"])
    return true_demand.rename(columns={"week_start": "target_week", "expected_units": "oracle"})


def score_rows(
    test: pd.DataFrame, model_pred: np.ndarray, true_demand: pd.DataFrame
) -> pd.DataFrame:
    """Attach every method's prediction to the test rows, keeping rows where all exist.

    Seasonal naive needs a year of history, so young outlets drop out. All methods are scored
    on the SAME rows, otherwise the comparison would not be fair.

    The oracle predicts the simulator's true mean demand. Nobody can know that in real life,
    so it shows the best any model could do: the rest of the error is pure noise and
    stock-outs. (Strictly, the WAPE-optimal forecast is the median, not the mean, so the
    oracle is a close practical floor rather than a mathematical proof.)
    """
    key = ["target_week", "outlet_id", "sku_id"]
    scored = test.assign(model=model_pred)
    for column in ("outlet_id", "sku_id"):  # categorical vs text keys would not merge cleanly
        scored[column] = scored[column].astype(str)
    scored = scored.merge(true_demand, on=key, how="left", validate="many_to_one")
    return scored[scored["seasonal_naive"].notna() & scored["oracle"].notna()]


def evaluation_table(scored: pd.DataFrame) -> pd.DataFrame:
    """MAPE / RMSE / WAPE for every method (model, baselines, oracle)."""
    rows = {
        name: score(scored["target_units"], scored[column])
        for name, column in PREDICTION_COLUMNS.items()
    }
    return pd.DataFrame(rows).T.round(2)


def wape_by(scored: pd.DataFrame, group_column: str) -> pd.DataFrame:
    """WAPE % of every method within each value of `group_column`, plus the row count."""
    rows = {}
    for value, group in scored.groupby(group_column, observed=True):
        actual = group["target_units"].to_numpy(float)
        rows[value] = {
            name: wape(actual, group[column].to_numpy(float))
            for name, column in PREDICTION_COLUMNS.items()
        }
        rows[value]["Test rows"] = len(group)
    table = pd.DataFrame(rows).T
    table["Test rows"] = table["Test rows"].astype(int)
    return table.round(2)


def horizon_table(scored: pd.DataFrame) -> pd.DataFrame:
    """WAPE % by forecast horizon (1 = next week ... 4 = four weeks ahead)."""
    table = wape_by(scored, "horizon").drop(columns="Test rows")
    table.index = [f"+{h}w" for h in table.index]
    return table


def to_markdown(table: pd.DataFrame) -> str:
    """Render a small DataFrame as a markdown table (for results/ and the README)."""

    def fmt(value: float) -> str:
        return f"{value:,}" if isinstance(value, (int, np.integer)) else f"{value:,.2f}"

    header = "| | " + " | ".join(table.columns) + " |"
    divider = "|---|" + "---|" * len(table.columns)
    lines = [
        f"| {name} | " + " | ".join(fmt(v) for v in row) + " |"
        for name, row in zip(table.index, table.itertuples(index=False), strict=True)
    ]
    return "\n".join([header, divider, *lines])


def main() -> None:
    model = joblib.load(MODELS_DIR / "lgbm_forecast.joblib")
    data = make_supervised(load_panel(sa.create_engine(DB_URL)))
    _, _, test = time_split(data)
    scored = score_rows(test, model.predict(test[FEATURES]), load_true_demand())
    print(f"Test period: {test['target_week'].min().date()} to {test['target_week'].max().date()}")
    print("\nOverall\n", evaluation_table(scored))
    print("\nWAPE % by horizon\n", horizon_table(scored))
    print("\nWAPE % by channel\n", wape_by(scored, "channel"))
    print("\nWAPE % by region\n", wape_by(scored, "region"))


if __name__ == "__main__":
    main()
