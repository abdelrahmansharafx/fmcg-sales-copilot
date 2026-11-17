"""Forecast metrics and the model-vs-baseline evaluation table.

Run on its own to re-score the saved model on the test period:
    uv run python -m src.forecast.evaluate
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import sqlalchemy as sa

from src.config import DB_URL, MODELS_DIR
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


def baseline_predictions(df: pd.DataFrame) -> dict[str, pd.Series]:
    """The two baselines the model has to beat.

    * Seasonal naive: units sold in the same week last year (undefined for young outlets).
    * Moving average: mean of the last 4 weeks. Not required, but a stronger yardstick,
      because last year's single noisy week is a weak forecast.
    """
    return {"Seasonal naive": df["seasonal_naive"], "Moving avg (4w)": df["mean_4"]}


def evaluation_table(test: pd.DataFrame, model_pred: np.ndarray) -> pd.DataFrame:
    """Compare model and baselines on the SAME rows (those where seasonal naive exists)."""
    test = test.assign(model=model_pred)
    comparable = test[test["seasonal_naive"].notna()]
    predictions = {**baseline_predictions(comparable), "LightGBM": comparable["model"]}
    rows = {name: score(comparable["target_units"], pred) for name, pred in predictions.items()}
    return pd.DataFrame(rows).T.round(2)


def horizon_table(test: pd.DataFrame, model_pred: np.ndarray) -> pd.DataFrame:
    """WAPE % by forecast horizon (1 = next week ... 4 = four weeks ahead)."""
    test = test.assign(model=model_pred)
    comparable = test[test["seasonal_naive"].notna()]
    columns = {
        "Seasonal naive": "seasonal_naive",
        "Moving avg (4w)": "mean_4",
        "LightGBM": "model",
    }
    rows = {}
    for horizon, group in comparable.groupby("horizon"):
        rows[f"+{horizon}w"] = {
            name: wape(group["target_units"].to_numpy(float), group[col].to_numpy(float))
            for name, col in columns.items()
        }
    return pd.DataFrame(rows).T.round(2)


def to_markdown(table: pd.DataFrame) -> str:
    """Render a small DataFrame as a markdown table (for results/ and the README)."""
    header = "| | " + " | ".join(table.columns) + " |"
    divider = "|---|" + "---|" * len(table.columns)
    lines = [
        f"| {name} | " + " | ".join(f"{v:.2f}" for v in row) + " |"
        for name, row in table.iterrows()
    ]
    return "\n".join([header, divider, *lines])


def main() -> None:
    model = joblib.load(MODELS_DIR / "lgbm_forecast.joblib")
    data = make_supervised(load_panel(sa.create_engine(DB_URL)))
    _, _, test = time_split(data)
    pred = model.predict(test[FEATURES])
    print(f"Test period: {test['target_week'].min().date()} to {test['target_week'].max().date()}")
    print("\nOverall (rows where seasonal naive exists)\n", evaluation_table(test, pred))
    print("\nWAPE % by horizon\n", horizon_table(test, pred))


if __name__ == "__main__":
    main()
