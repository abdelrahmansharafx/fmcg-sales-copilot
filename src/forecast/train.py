"""Train the LightGBM demand model, compare it with the baselines and log the run to MLflow.

    uv run python -m src.forecast.train

Outputs: models/lgbm_forecast.joblib, models/model_meta.json, results/forecast_metrics.md and
an MLflow run in ./mlruns (browse it with `uv run mlflow ui`).
"""

from __future__ import annotations

import json

import joblib
import lightgbm as lgb
import mlflow
import pandas as pd
import sqlalchemy as sa

from src.config import DB_URL, MLFLOW_TRACKING_URI, MODELS_DIR, RESULTS_DIR
from src.forecast.evaluate import (
    ORACLE,
    evaluation_table,
    horizon_table,
    load_true_demand,
    score,
    score_rows,
    to_markdown,
    wape_by,
)
from src.forecast.features import (
    CATEGORICAL_FEATURES,
    FEATURES,
    load_panel,
    make_supervised,
    time_split,
)

SEED = 42
PARAMS = {
    # Units are counts (0, 1, 2, ...), so a Poisson loss fits better than plain squared error:
    # it keeps predictions positive and treats errors relative to the volume level.
    "objective": "poisson",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 100,  # each leaf must cover enough rows: fights overfitting
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "n_estimators": 2000,  # upper limit; early stopping picks the real number
    "random_state": SEED,
    "verbose": -1,
}
EARLY_STOPPING_ROUNDS = 50


def fit_model(train: pd.DataFrame, val: pd.DataFrame) -> lgb.LGBMRegressor:
    """Fit on the training period; the validation period decides when to stop adding trees."""
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(
        train[FEATURES],
        train["target_units"],
        eval_set=[(val[FEATURES], val["target_units"])],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS), lgb.log_evaluation(100)],
    )
    return model


def feature_importance(model: lgb.LGBMRegressor) -> pd.DataFrame:
    """Total loss reduction ('gain') each feature contributed, as a share of the total."""
    gain = model.booster_.feature_importance(importance_type="gain")
    table = pd.DataFrame({"feature": FEATURES, "gain_share": gain / gain.sum()})
    return table.sort_values("gain_share", ascending=False).reset_index(drop=True)


def main() -> None:
    engine = sa.create_engine(DB_URL)
    data = make_supervised(load_panel(engine))
    train, val, test = time_split(data)
    splits = {"train": train, "val": val, "test": test}
    periods = {
        name: f"{df['target_week'].min().date()} to {df['target_week'].max().date()}"
        for name, df in splits.items()
    }
    for name, df in splits.items():
        print(f"{name:>5}: {len(df):>9,} rows, targets {periods[name]}")

    model = fit_model(train, val)

    # The test set is used exactly once, below, after every modelling decision is made.
    test_pred = model.predict(test[FEATURES])
    scored = score_rows(test, test_pred, load_true_demand())
    overall = evaluation_table(scored)
    by_horizon = horizon_table(scored)
    by_channel = wape_by(scored, "channel")
    by_region = wape_by(scored, "region")
    all_rows = score(test["target_units"], test_pred)  # includes young outlets (no baseline)
    importance = feature_importance(model)

    print(f"\nTest period: {periods['test']}")
    print("\nOverall, test period (rows where seasonal naive exists)\n", overall)
    print("\nWAPE % by horizon\n", by_horizon)
    print("\nWAPE % by channel\n", by_channel)
    print("\nWAPE % by region\n", by_region)
    print("\nTop features\n", importance.head(8).to_string(index=False))

    # --- save the model and what Phase 3 needs to use it correctly --------------------------
    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    model_path = MODELS_DIR / "lgbm_forecast.joblib"
    joblib.dump(model, model_path)
    meta = {
        "features": FEATURES,
        "categorical_levels": {c: list(data[c].cat.categories) for c in CATEGORICAL_FEATURES},
        "best_iteration": model.best_iteration_,
        "test_wape_pct": float(overall.loc["LightGBM", "WAPE %"]),
        "periods": periods,
    }
    (MODELS_DIR / "model_meta.json").write_text(json.dumps(meta, indent=2))

    report = (
        f"## Forecast results (test period {periods['test']})\n\n"
        "Time-based split; all methods scored on the same test rows (those where the "
        "seasonal-naive baseline exists). Lower is better.\n\n"
        f"{to_markdown(overall)}\n\n"
        "The oracle predicts the simulator's true mean demand, which no real model can know: "
        "what is left is pure noise and stock-outs.\n\n"
        f"### WAPE % by horizon\n\n{to_markdown(by_horizon)}\n\n"
        f"### WAPE % by channel\n\n{to_markdown(by_channel)}\n\n"
        f"### WAPE % by region\n\n{to_markdown(by_region)}\n"
    )
    report_path = RESULTS_DIR / "forecast_metrics.md"
    report_path.write_text(report)
    importance_path = RESULTS_DIR / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)

    # --- log everything to MLflow ----------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("demand-forecast")
    with mlflow.start_run(run_name="lgbm-poisson-direct-4w"):
        mlflow.log_params({**PARAMS, "n_features": len(FEATURES), "train_rows": len(train)})
        mlflow.log_params({f"{name}_period": period for name, period in periods.items()})
        mlflow.log_metric("best_iteration", model.best_iteration_)
        metric_keys = {
            "Seasonal naive": "seasonal_naive",
            "Moving avg (4w)": "moving_avg_4w",
            "LightGBM": "lightgbm",
            ORACLE: "oracle",
        }
        for model_name, row in overall.iterrows():
            prefix = metric_keys[model_name]
            for metric, value in row.items():
                mlflow.log_metric(f"test_{prefix}_{metric.split()[0].lower()}", value)
        for metric, value in all_rows.items():
            mlflow.log_metric(f"test_lightgbm_all_rows_{metric.split()[0].lower()}", value)
        for path in (model_path, MODELS_DIR / "model_meta.json", report_path, importance_path):
            mlflow.log_artifact(str(path))
    print(f"\nSaved {model_path} and logged run to {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()
