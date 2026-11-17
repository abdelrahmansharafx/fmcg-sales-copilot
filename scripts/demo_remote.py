"""Run the demo questions against a deployed copy of the API (e.g. on Azure Container Apps).

    uv run python scripts/demo_remote.py

It needs DEMO_URL (https://...) and DEMO_API_KEY, from the environment or from a `.env.demo` file
in the project root (git-ignored, so the key never ends up in git, shell history or chat).
The key is never printed. Exit code 1 if any check fails.
"""

from __future__ import annotations

import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
Out = Callable[[str], None]


@dataclass
class Step:
    """One demo call. `expect` are regexes that must ALL match; `forbid` must NOT match."""

    name: str
    method: str
    path: str
    body: dict | None = None
    expect: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()


ASK = "/ask"
STEPS = [
    Step("REST: forecast O001 / S01, 4 weeks", "POST", "/forecast",
         {"outlet_id": "O001", "sku_id": "S01", "weeks": 4}, expect=(r'"total_units"',)),
    Step("REST: order recommendation for O010", "GET", "/recommend/O010",
         expect=(r'"total_order_units"', r'"Coca-Cola')),
    Step("Agent: forecast", "POST", ASK,
         {"question": "Forecast Coca-Cola Original 330ml can at outlet O001 for the next 4 weeks."},
         expect=(r"\d{3}", r"(?i)error|estimate|typical")),
    Step("Agent: order recommendation", "POST", ASK,
         {"question": "What should outlet O010 order next week? Show the top 5 lines."},
         expect=(r"(?i)coca-cola",)),
    Step("Agent: promotion impact", "POST", ASK,
         {"question": "Did promotion 2 on Coca-Cola Original 330ml can pay off?"},
         expect=(r"35\.\d\s*%", r"(?i)confidence|CI")),
    Step("Agent: policy (RAG)", "POST", ASK,
         {"question": "Can I run a 30% discount, and how far ahead must I request a promotion?"},
         expect=(r"(?i)3 weeks|three weeks", r"(?i)commercial director")),
    Step("Agent: sales question (text-to-SQL)", "POST", ASK,
         {"question": "Which Alexandria outlet had the largest percentage drop in units over "
                      "the last 4 weeks compared with the 4 weeks before?"},
         expect=(r"(?i)cafe 115|O115",)),
    Step("Agent: unanswerable (must say it does not know)", "POST", ASK,
         {"question": "What was our market share versus Pepsi last quarter?"},
         expect=(r"(?i)don't know|do not know|cannot|no competitor|not (available|contain)",),
         forbid=(r"\d+(\.\d+)?\s*%",)),
]  # fmt: skip


def load_settings() -> tuple[str, str]:
    """DEMO_URL and DEMO_API_KEY from the environment, else from .env.demo."""
    file_values = dotenv_values(ROOT / ".env.demo")
    url = os.getenv("DEMO_URL") or file_values.get("DEMO_URL") or ""
    key = os.getenv("DEMO_API_KEY") or file_values.get("DEMO_API_KEY") or ""
    if not url or not key:
        sys.exit(
            "Set DEMO_URL and DEMO_API_KEY (environment or .env.demo). See guides/azure_deploy.md."
        )
    return url.rstrip("/"), key


def wait_until_ready(
    client, out: Out, timeout: float = 300, sleep: Callable[[float], None] = time.sleep
) -> float:
    """Poll /health until 200. A scale-to-zero app needs a cold start; returns seconds waited."""
    started = time.perf_counter()
    while True:
        try:
            response = client.get("/health")
            if response.status_code == 200:
                waited = time.perf_counter() - started
                out(f"Ready after {waited:.1f}s: {response.json()}")
                return waited
            out(f"  /health -> HTTP {response.status_code}, waiting...")
        except httpx.HTTPError as error:
            out(f"  not reachable yet ({type(error).__name__}), waiting...")
        if time.perf_counter() - started > timeout:
            raise TimeoutError(f"The API was not ready after {timeout:.0f}s")
        sleep(5)


def check_auth(client, out: Out) -> bool:
    """Protected endpoints must refuse requests without a valid key."""
    body = {"outlet_id": "O001", "sku_id": "S01"}
    no_key = client.post("/forecast", json=body).status_code
    wrong = client.post("/forecast", json=body, headers={"X-API-Key": "wrong"}).status_code
    ok = no_key == 401 and wrong == 401
    out(f"{'PASS' if ok else 'FAIL'}  auth is enforced (no key -> {no_key}, wrong key -> {wrong})")
    return ok


def run_step(client, key: str, step: Step, out: Out) -> dict:
    """Run one step and print what happened. Returns a result dict."""
    started = time.perf_counter()
    response = client.request(step.method, step.path, json=step.body, headers={"X-API-Key": key})
    seconds = time.perf_counter() - started
    data = (
        response.json()
        if response.headers.get("content-type", "").startswith("application/json")
        else {}
    )
    text = data.get("answer") if step.path == ASK else response.text
    text = text or ""

    problems = []
    if response.status_code != 200:
        problems.append(f"HTTP {response.status_code}: {response.text[:200]}")
    problems += [f"missing /{p}/" for p in step.expect if not re.search(p, text)]
    problems += [f"forbidden /{p}/ matched" for p in step.forbid if re.search(p, text)]

    out(f"\n{'PASS' if not problems else 'FAIL'}  {step.name}  ({seconds:.1f}s)")
    if step.path == ASK and response.status_code == 200:
        tools = [t["name"] for t in data["tool_calls"]] or ["none"]
        usage = data["usage"]
        out(f"      tools: {', '.join(tools)}")
        out(f"      tokens: {usage['prompt_tokens'] + usage['completion_tokens']:,}  "
            f"estimated cost: ${usage['cost_usd']:.4f}")  # fmt: skip
        out("      " + " ".join(text.split())[:520])
    for problem in problems:
        out(f"      ! {problem}")
    return {
        "name": step.name,
        "ok": not problems,
        "seconds": seconds,
        "cost": data.get("usage", {}).get("cost_usd", 0.0) if step.path == ASK else 0.0,
        "is_ask": step.path == ASK,
    }


def run_demo(client, key: str, out: Out = print, steps: list[Step] | None = None) -> list[dict]:
    """The whole demo. `client` is an httpx.Client (or a FastAPI TestClient) with base_url set."""
    wait_until_ready(client, out)
    results = [
        {"name": "auth", "ok": check_auth(client, out), "seconds": 0, "cost": 0, "is_ask": False}
    ]
    for step in steps or STEPS:
        results.append(run_step(client, key, step, out))

    asks = [r for r in results if r["is_ask"]]
    passed = sum(r["ok"] for r in results)
    out(f"\n{passed}/{len(results)} checks passed")
    if asks:
        average = sum(r["seconds"] for r in asks) / len(asks)
        total_cost = sum(r["cost"] for r in asks)
        out(f"/ask: average latency {average:.1f}s, estimated total cost ${total_cost:.4f} "
            f"for {len(asks)} questions")  # fmt: skip
    return results


def main() -> int:
    url, key = load_settings()
    print(f"Demo against {url}")
    with httpx.Client(base_url=url, timeout=180) as client:  # a cold start can take a while
        results = run_demo(client, key)
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
