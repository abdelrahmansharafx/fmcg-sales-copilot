## Forecast results (test period 2025-09-29 to 2025-12-22)

Time-based split; metrics on test rows where the seasonal-naive baseline exists. Lower is better.

| | MAPE % | RMSE | WAPE % |
|---|---|---|---|
| Seasonal naive | 59.23 | 59.44 | 47.36 |
| Moving avg (4w) | 56.15 | 50.26 | 40.70 |
| LightGBM | 47.30 | 42.65 | 33.91 |

### WAPE % by horizon

| | Seasonal naive | Moving avg (4w) | LightGBM |
|---|---|---|---|
| +1w | 47.36 | 38.14 | 33.21 |
| +2w | 47.36 | 39.89 | 33.87 |
| +3w | 47.36 | 41.54 | 34.12 |
| +4w | 47.36 | 43.23 | 34.43 |
