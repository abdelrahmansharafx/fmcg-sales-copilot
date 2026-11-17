# Evaluation results

Run on 2026-11-17 12:54 UTC with provider `azure`, deployment `copilot-chat` (model `gpt-4.1-mini-2025-04-14`),
37 questions, temperature 0. LLM judge: on (same model as the agent).

## Summary

| Category | Questions | Tool choice | Answer correct | Grounded (RAG) | All passed | Avg latency | Avg tokens | Avg cost |
|---|---|---|---|---|---|---|---|---|
| **All** | 37 | 100% (37/37) | 100% (37/37) | 100% (10/10) | 100% (37/37) | 5.3s | 4,979 | $0.0022 |
| error-handling | 1 | 100% (1/1) | 100% (1/1) | n/a | 100% (1/1) | 2.6s | 4,245 | $0.0018 |
| forecast | 2 | 100% (2/2) | 100% (2/2) | n/a | 100% (2/2) | 2.7s | 4,392 | $0.0019 |
| multi-tool | 4 | 100% (4/4) | 100% (4/4) | 100% (3/3) | 100% (4/4) | 5.1s | 5,467 | $0.0024 |
| policy | 7 | 100% (7/7) | 100% (7/7) | 100% (7/7) | 100% (7/7) | 5.9s | 5,357 | $0.0023 |
| promo | 5 | 100% (5/5) | 100% (5/5) | n/a | 100% (5/5) | 6.6s | 5,733 | $0.0025 |
| recommend | 3 | 100% (3/3) | 100% (3/3) | n/a | 100% (3/3) | 3.2s | 5,049 | $0.0022 |
| refusal | 6 | 100% (6/6) | 100% (6/6) | n/a | 100% (6/6) | 2.4s | 3,124 | $0.0014 |
| sql | 9 | 100% (9/9) | 100% (9/9) | n/a | 100% (9/9) | 7.9s | 5,474 | $0.0025 |

- **Tool choice**: the tools called include every expected tool and nothing outside the expected/allowed set.
- **Answer correct**: all checks of the question pass. Numbers and names are checked deterministically against values computed from the database or by calling the tool directly; free text is graded by an LLM judge against key facts.
- **Grounded (RAG)**: policy answers only claim what the retrieved passages say (LLM judge). A policy answer where `search_policies` was not used counts as not grounded.
- **All passed**: tool choice, correctness and groundedness all OK.

## Original vs fresh questions

The prompt and the judge were improved after the first run exposed failures on the original questions. The fresh variants were written before the re-run and never used for tuning, so they show whether the fixes generalise.

| Question set | Questions | Tool choice | Answer correct | Grounded (RAG) | All passed | Avg latency | Avg tokens | Avg cost |
|---|---|---|---|---|---|---|---|---|
| Original questions (used while tuning) | 31 | 100% (31/31) | 100% (31/31) | 100% (9/9) | 100% (31/31) | 5.2s | 4,972 | $0.0022 |
| Fresh variants (not tuned on) | 6 | 100% (6/6) | 100% (6/6) | 100% (1/1) | 100% (6/6) | 6.0s | 5,016 | $0.0022 |

## Cost and latency

| Metric | Value |
|---|---|
| Average latency per question | 5.3s |
| Median / 95th percentile latency | 4.1s / 16.2s |
| Average tokens per question (prompt + completion) | 4,833 + 146 = 4,979 |
| Average estimated cost per question | $0.0022 |
| Most expensive question | reg-02 ($0.0035) |
| Total estimated cost of the 37 questions | $0.080 |
| Judge overhead (eval only, not a per-question product cost) | 28 calls, 16,674 tokens, $0.010 |

Cost is an **estimate**: tokens x list price of $0.40 / $1.60 per 1M input / output tokens (override with `LLM_PRICE_INPUT_PER_1M` / `LLM_PRICE_OUTPUT_PER_1M`). Tokens include every LLM call of the question: the agent's own calls plus the text-to-SQL calls made inside `query_sales`. Latency is wall-clock, one question at a time, and includes network, tools and the LLM.

## Regression cases

Failures we hit earlier in development and fixed; they must keep passing.

| Question | Result |
|---|---|
| `reg-02`: Can I run a 30% discount on Sprite 1L, and how far in advance do I have to request a promotion? | PASS |
| `reg-01`: What was our market share versus Pepsi in Cairo last quarter? | PASS |
| `reg-03`: Which Alexandria outlet had the largest percentage drop in units over the last 4 weeks compared with the 4 weeks before? | PASS |

## Failures

No failures.

## Per-question detail

| ID | Tools called | Tool | Correct | Grounded | Latency | Tokens | Cost |
|---|---|---|---|---|---|---|---|
| sql-01 | query_sales | yes | yes | - | 16.5s | 5,008 | $0.0021 |
| sql-02 | query_sales | yes | yes | - | 5.3s | 5,114 | $0.0022 |
| sql-03 | query_sales | yes | yes | - | 8.8s | 5,181 | $0.0023 |
| sql-04 | query_sales | yes | yes | - | 5.5s | 5,208 | $0.0023 |
| sql-05 | query_sales | yes | yes | - | 6.6s | 5,268 | $0.0023 |
| sql-06 | query_sales | yes | yes | - | 5.2s | 5,137 | $0.0022 |
| sql-07 | query_sales | yes | yes | - | 6.7s | 5,770 | $0.0028 |
| sql-08 | query_sales | yes | yes | - | 7.5s | 5,884 | $0.0029 |
| fc-01 | forecast_demand | yes | yes | - | 2.7s | 4,428 | $0.0019 |
| fc-02 | forecast_demand | yes | yes | - | 2.7s | 4,356 | $0.0019 |
| fc-03 | forecast_demand | yes | yes | - | 2.6s | 4,245 | $0.0018 |
| rec-01 | recommend_order | yes | yes | - | 2.9s | 5,022 | $0.0021 |
| rec-02 | recommend_order | yes | yes | - | 3.1s | 5,019 | $0.0021 |
| rec-03 | recommend_order | yes | yes | - | 3.5s | 5,107 | $0.0023 |
| pr-01 | promo_impact | yes | yes | - | 2.9s | 4,447 | $0.0019 |
| pr-02 | promo_impact | yes | yes | - | 2.9s | 4,446 | $0.0019 |
| pr-03 | query_sales, promo_impact | yes | yes | - | 7.0s | 7,608 | $0.0033 |
| pol-01 | search_policies | yes | yes | yes | 3.1s | 4,784 | $0.0020 |
| pol-02 | search_policies | yes | yes | yes | 3.7s | 4,828 | $0.0020 |
| pol-03 | search_policies | yes | yes | yes | 2.5s | 4,785 | $0.0020 |
| pol-04 | search_policies | yes | yes | yes | 5.6s | 5,041 | $0.0021 |
| pol-05 | search_policies | yes | yes | yes | 6.3s | 4,870 | $0.0021 |
| pol-06 | search_policies | yes | yes | yes | 4.2s | 4,981 | $0.0021 |
| reg-02 | search_policies, search_policies | yes | yes | yes | 16.1s | 8,213 | $0.0035 |
| mt-01 | promo_impact, search_policies | yes | yes | yes | 5.1s | 5,170 | $0.0023 |
| mt-02 | search_policies, recommend_order | yes | yes | yes | 4.5s | 6,031 | $0.0026 |
| reg-01 | none | yes | yes | - | 2.4s | 2,124 | $0.0009 |
| ref-01 | search_policies | yes | yes | - | 3.5s | 5,145 | $0.0022 |
| ref-02 | none | yes | yes | - | 2.0s | 2,106 | $0.0009 |
| ref-03 | none | yes | yes | - | 1.4s | 2,103 | $0.0009 |
| v-01 | query_sales, promo_impact | yes | yes | - | 17.0s | 7,716 | $0.0033 |
| v-02 | promo_impact | yes | yes | - | 3.4s | 4,448 | $0.0019 |
| v-03 | promo_impact, search_policies | yes | yes | yes | 4.1s | 5,173 | $0.0023 |
| v-04 | search_policies | yes | yes | - | 3.5s | 5,160 | $0.0022 |
| v-05 | none | yes | yes | - | 1.5s | 2,105 | $0.0009 |
| v-06 | forecast_demand, query_sales | yes | yes | - | 6.8s | 5,495 | $0.0025 |
| reg-03 | query_sales | yes | yes | - | 8.5s | 6,693 | $0.0033 |

## History of this eval

Each step below changed the agent or the eval after seeing results, so the numbers are not directly comparable.

| Run | Questions | Passed | What happened |
|---|---|---|---|
| 1 | 30 | 25 (83%) | First run. 3 real agent failures: uplift worked out with SQL instead of `promo_impact` (`pr-03`); a different promotion analysed than the one the user named (`mt-01`); the retailer margin cap given as "our profit margin" (`ref-01`). 2 judge failures: extra detail penalised (`rec-03`) and non-policy claims flagged as ungrounded (`mt-02`). |
| 2 | 36 | 31 (86%) | Agent prompt tightened (always use `promo_impact` for uplift, use ids as given, no related-but-different figures, answer every part). Groundedness judge now must quote evidence, verified in code. 6 fresh variants added. Remaining failures were mostly the judge failing multi-part answers and two over-strict "no % anywhere" regexes on the margin questions. |
| 3 | 36 | 35 (97%) | Correctness judge redesigned as a checklist with quotes verified in code. The two margin regexes were removed (the agent now correctly says the retailer margin cap is not our margin). Rate-limit retries added after Azure returned HTTP 429 mid-run. The one failure (`ref-03`) was a bug in the eval: a conditional key fact ("any number is labelled an estimate") can never be present when no number is given. |
| 4 | 36 | 36 (100%) | After fixing the `ref-03` key fact. |
| 5 | 37 | see Summary | Phase 6. Demoing `/ask` showed a text-to-SQL bug the evals had missed: for the wording "...compared with the 4 weeks before?" the LLM wrote the previous window as `BETWEEN -56 AND -29 days`, so the week at exactly -28 days fell in neither window and outlets were ranked on non-adjacent periods (it named Kiosk 243 instead of Cafe 115). The `query_sales` prompt now gives exact half-open conditions and forbids `BETWEEN` for week windows. `reg-03` (the failing wording) was added as a regression test. |

Honest reading: the evals passed while this bug existed because `sql-07` used slightly different wording, which shows how sensitive text-to-SQL is to phrasing. Questions were fixed after they failed, so the original 30 are partly tuned on; the 6 fresh variants were written before run 2 and not tuned on. With 36 questions, a difference of one question is within noise.

## Method and limitations

- Expected answers for data questions are computed when the eval runs (`truth_sql`, or a direct call of the tool), so they cannot go stale if the data is regenerated. Tool-based questions check that the agent reports the tool's numbers faithfully; the tools themselves are unit-tested separately.
- The judge is an LLM from the same family as the agent, so it can be lenient; regex checks back it up where the key fact is a simple phrase.
- One run per question at temperature 0. LLM output can still vary slightly between runs, and 30 questions is a small sample: treat a 1-2 question change as noise.
- Number matching reads any number in the answer; a coincidental match is possible for small expected values.
- Not measured yet: whether every number in an answer appears in a tool result (a general "no invented numbers" check).
