# Using the API

```bash
# one-off: put a long random key in .env
python3 -c "import secrets; print('API_KEY=' + secrets.token_urlsafe(32))" >> .env

uv run python -m src.api                      # http://127.0.0.1:8000  (interactive docs at /docs)
```

All endpoints except `/health` and the chat page need the header `X-API-Key: <your key>`.

## The chat page

Open `http://127.0.0.1:8000/` (or your deployed URL) for a simple chat UI served by the same app: paste
your API key once (it is kept in your browser's localStorage and sent only in the `X-API-Key` header to
this server), then ask a question or click one of the five examples. Each answer shows the tools used,
the latency and the estimated cost. If the server has been idle and is waking up, the page tells you.

- The page itself is public and contains no data; only `/ask` needs the key.
- A content-security-policy (`connect-src 'self'`) stops the page from sending anything, including the
  key, to any other site, and answers are rendered as plain text, never as HTML.
- Limitation: if the app has scaled to zero, the *first load of the page* waits for the container to
  start (about 45 s) before anything appears, because the page is served by that same container. The
  "waking up" message covers requests made after the page is open.

```bash
KEY=...   # the value of API_KEY

curl localhost:8000/health

curl -X POST localhost:8000/ask -H "X-API-Key: $KEY" -H 'content-type: application/json' \
  -d '{"question": "Which Alexandria outlet had the largest percentage drop in units over the last 4 weeks vs the 4 before?"}'

curl -X POST localhost:8000/forecast -H "X-API-Key: $KEY" -H 'content-type: application/json' \
  -d '{"outlet_id": "O001", "sku_id": "S01", "weeks": 4}'

curl localhost:8000/recommend/O010 -H "X-API-Key: $KEY"
```

| Status | Meaning |
|---|---|
| 401 | Missing or wrong `X-API-Key` |
| 404 | Unknown outlet / SKU / promotion, or the outlet does not stock that SKU |
| 422 | Invalid input (weeks outside 1-4, malformed id, empty question) |
| 429 / 502 | The LLM is rate limited / unreachable (`/ask` only) |
| 503 | Not ready: no `API_KEY` configured, model not trained, database missing, or no LLM key |

## What each request logs

One JSON line per request on stdout, for example:

```json
{"ts": "...", "level": "INFO", "logger": "fmcg.api", "message": "request", "event": "request",
 "request_id": "demo-req-001", "method": "POST", "path": "/ask", "status": 200,
 "latency_ms": 9381.2, "llm_calls": 3, "prompt_tokens": 5335, "completion_tokens": 570,
 "cost_usd": 0.003046, "api_key_id": "78d6e773", "tools": ["query_sales"], "question_chars": 105}
```

- `cost_usd` is an **estimate** (tokens x list price; see `LLM_PRICE_*` in `.env.example`). It is 0 for
  `/forecast` and `/recommend` because they do not call an LLM.
- Send `X-Request-ID` to correlate a request with its log line; otherwise one is generated and returned.
- The key itself and the question text are never logged: only a short key fingerprint (`api_key_id`)
  and the question length.
- `/health` is logged at DEBUG only, so load-balancer probes do not flood the log.
