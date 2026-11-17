# Deployment measurements

Public API on Azure Container Apps (Sweden Central), measured with `scripts/demo_remote.py`
from a laptop on 2026-11-17. Synthetic data only.

| Setting | Value |
|---|---|
| Plan | Container Apps Consumption, 1 vCPU / 2 GiB |
| Scaling | min replicas 0 (scale to zero), max replicas 1 |
| Image | about 342 MB compressed (1.35 GB unpacked), linux/amd64, runs as a non-root user |
| LLM | Azure OpenAI deployment `copilot-chat` (model `gpt-4.1-mini`) |

| Measurement | Result |
|---|---|
| **Cold start** (first `/health` after 10+ minutes idle, app scaled to zero) | **47.4 s** (one measurement) |
| Warm `/health` | under 1 s (0.4 s measured) |
| Warm REST endpoints (`/forecast`, `/recommend/{id}`) | 0.1 to 0.2 s |
| `/ask` average latency, 6 questions, warm | 5.0 s (run 1), 4.6 s (run 2, straight after the cold start) |
| `/ask` estimated cost | about $0.002 per question ($0.0123 for the 6 demo questions) |
| Demo checks | 9 of 9 passed on both runs |

Notes
- The cold start is one sample and bundles three things: starting a replica, pulling the image
  from the registry, and the app loading its model and policy index. It is not split further.
- Latency is wall-clock from a laptop and includes the network and the Azure OpenAI calls.
- Cost is an estimate from token counts at list prices (see `LLM_PRICE_*` in `.env.example`).
- For a live demo, call `/health` a minute beforehand to wake the app, or set min replicas to 1
  for the day.
