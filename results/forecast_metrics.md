## Forecast results (test period 2025-09-29 to 2025-12-22)

Time-based split; all methods scored on the same test rows (those where the seasonal-naive baseline exists). Lower is better.

| | MAPE % | RMSE | WAPE % |
|---|---|---|---|
| Seasonal naive | 59.23 | 59.44 | 47.36 |
| Moving avg (4w) | 56.15 | 50.26 | 40.70 |
| LightGBM | 47.30 | 42.65 | 33.91 |
| Oracle (true mean), unreachable floor | 42.91 | 40.96 | 32.36 |

The oracle predicts the simulator's true mean demand, which no real model can know: what is left is pure noise and stock-outs.

### WAPE % by horizon

| | Seasonal naive | Moving avg (4w) | LightGBM | Oracle (true mean), unreachable floor |
|---|---|---|---|---|
| +1w | 47.36 | 38.14 | 33.21 | 32.36 |
| +2w | 47.36 | 39.89 | 33.87 | 32.36 |
| +3w | 47.36 | 41.54 | 34.12 | 32.36 |
| +4w | 47.36 | 43.23 | 34.43 | 32.36 |

### WAPE % by channel

| | Seasonal naive | Moving avg (4w) | LightGBM | Oracle (true mean), unreachable floor | Test rows |
|---|---|---|---|---|---|
| hypermarket | 45.42 | 39.28 | 32.60 | 31.07 | 35,256 |
| kiosk | 51.03 | 43.15 | 36.49 | 34.50 | 51,236 |
| restaurant_cafe | 49.27 | 41.53 | 34.90 | 33.71 | 24,244 |
| supermarket | 48.63 | 41.87 | 34.80 | 33.29 | 68,744 |

### WAPE % by region

| | Seasonal naive | Moving avg (4w) | LightGBM | Oracle (true mean), unreachable floor | Test rows |
|---|---|---|---|---|---|
| Alexandria | 47.28 | 40.72 | 33.88 | 32.04 | 24,832 |
| Cairo | 47.16 | 39.87 | 33.34 | 31.97 | 59,592 |
| Delta | 47.79 | 40.00 | 33.91 | 32.36 | 36,748 |
| Giza | 48.13 | 41.10 | 34.62 | 33.58 | 32,360 |
| Upper Egypt | 46.59 | 43.13 | 34.58 | 32.33 | 25,948 |
