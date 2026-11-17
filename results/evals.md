# Evaluation results

Run on 2026-11-17 12:33 UTC with provider `azure`, deployment `copilot-chat` (model `gpt-4.1-mini-2025-04-14`),
36 questions, temperature 0. LLM judge: on (same model as the agent).

## Summary

| Category | Questions | Tool choice | Answer correct | Grounded (RAG) | All passed | Avg latency | Avg tokens | Avg cost |
|---|---|---|---|---|---|---|---|---|
| **All** | 36 | 100% (36/36) | 100% (36/36) | 100% (10/10) | 100% (36/36) | 6.3s | 4,872 | $0.0021 |
| error-handling | 1 | 100% (1/1) | 100% (1/1) | n/a | 100% (1/1) | 8.2s | 4,246 | $0.0018 |
| forecast | 2 | 100% (2/2) | 100% (2/2) | n/a | 100% (2/2) | 5.4s | 4,396 | $0.0019 |
| multi-tool | 4 | 100% (4/4) | 100% (4/4) | 100% (3/3) | 100% (4/4) | 7.0s | 5,500 | $0.0024 |
| policy | 7 | 100% (7/7) | 100% (7/7) | 100% (7/7) | 100% (7/7) | 5.3s | 5,234 | $0.0022 |
| promo | 5 | 100% (5/5) | 100% (5/5) | n/a | 100% (5/5) | 9.6s | 5,668 | $0.0024 |
| recommend | 3 | 100% (3/3) | 100% (3/3) | n/a | 100% (3/3) | 9.1s | 5,044 | $0.0022 |
| refusal | 6 | 100% (6/6) | 100% (6/6) | n/a | 100% (6/6) | 3.1s | 3,120 | $0.0013 |
| sql | 8 | 100% (8/8) | 100% (8/8) | n/a | 100% (8/8) | 6.2s | 5,192 | $0.0023 |

- **Tool choice**: the tools called include every expected tool and nothing outside the expected/allowed set.
- **Answer correct**: all checks of the question pass. Numbers and names are checked deterministically against values computed from the database or by calling the tool directly; free text is graded by an LLM judge against key facts.
- **Grounded (RAG)**: policy answers only claim what the retrieved passages say (LLM judge). A policy answer where `search_policies` was not used counts as not grounded.
- **All passed**: tool choice, correctness and groundedness all OK.

## Original vs fresh questions

The prompt and the judge were improved after the first run exposed failures on the original questions. The fresh variants were written before the re-run and never used for tuning, so they show whether the fixes generalise.

| Question set | Questions | Tool choice | Answer correct | Grounded (RAG) | All passed | Avg latency | Avg tokens | Avg cost |
|---|---|---|---|---|---|---|---|---|
| Original questions (used while tuning) | 30 | 100% (30/30) | 100% (30/30) | 100% (9/9) | 100% (30/30) | 6.1s | 4,846 | $0.0021 |
| Fresh variants (not tuned on) | 6 | 100% (6/6) | 100% (6/6) | 100% (1/1) | 100% (6/6) | 7.5s | 5,005 | $0.0022 |

## Cost and latency

| Metric | Value |
|---|---|
| Average latency per question | 6.3s |
| Median / 95th percentile latency | 5.6s / 11.8s |
| Average tokens per question (prompt + completion) | 4,734 + 139 = 4,872 |
| Average estimated cost per question | $0.0021 |
| Most expensive question | v-01 ($0.0033) |
| Total estimated cost of the 36 questions | $0.076 |
| Judge overhead (eval only, not a per-question product cost) | 28 calls, 16,235 tokens, $0.009 |

Cost is an **estimate**: tokens x list price of $0.40 / $1.60 per 1M input / output tokens (override with `LLM_PRICE_INPUT_PER_1M` / `LLM_PRICE_OUTPUT_PER_1M`). Tokens include every LLM call of the question: the agent's own calls plus the text-to-SQL calls made inside `query_sales`. Latency is wall-clock, one question at a time, and includes network, tools and the LLM.

## Regression cases

Failures we hit earlier in development and fixed; they must keep passing.

| Question | Result |
|---|---|
| `reg-02`: Can I run a 30% discount on Sprite 1L, and how far in advance do I have to request a promotion? | PASS |
| `reg-01`: What was our market share versus Pepsi in Cairo last quarter? | PASS |

## Failures

No failures.

## Per-question detail

| ID | Tools called | Tool | Correct | Grounded | Latency | Tokens | Cost |
|---|---|---|---|---|---|---|---|
| sql-01 | query_sales | yes | yes | - | 4.9s | 4,868 | $0.0020 |
| sql-02 | query_sales | yes | yes | - | 5.7s | 4,999 | $0.0021 |
| sql-03 | query_sales | yes | yes | - | 5.9s | 5,023 | $0.0022 |
| sql-04 | query_sales | yes | yes | - | 5.6s | 5,060 | $0.0022 |
| sql-05 | query_sales | yes | yes | - | 5.6s | 5,114 | $0.0023 |
| sql-06 | query_sales | yes | yes | - | 5.0s | 4,990 | $0.0021 |
| sql-07 | query_sales | yes | yes | - | 9.1s | 5,725 | $0.0029 |
| sql-08 | query_sales | yes | yes | - | 7.6s | 5,755 | $0.0028 |
| fc-01 | forecast_demand | yes | yes | - | 3.1s | 4,435 | $0.0019 |
| fc-02 | forecast_demand | yes | yes | - | 7.8s | 4,356 | $0.0019 |
| fc-03 | forecast_demand | yes | yes | - | 8.2s | 4,246 | $0.0018 |
| rec-01 | recommend_order | yes | yes | - | 5.1s | 5,005 | $0.0021 |
| rec-02 | recommend_order | yes | yes | - | 9.9s | 5,024 | $0.0021 |
| rec-03 | recommend_order | yes | yes | - | 12.2s | 5,104 | $0.0023 |
| pr-01 | promo_impact | yes | yes | - | 5.2s | 4,442 | $0.0019 |
| pr-02 | promo_impact | yes | yes | - | 8.5s | 4,445 | $0.0019 |
| pr-03 | query_sales, promo_impact | yes | yes | - | 14.3s | 7,444 | $0.0032 |
| pol-01 | search_policies | yes | yes | yes | 8.7s | 4,784 | $0.0020 |
| pol-02 | search_policies | yes | yes | yes | 4.2s | 4,828 | $0.0020 |
| pol-03 | search_policies | yes | yes | yes | 4.6s | 4,785 | $0.0020 |
| pol-04 | search_policies | yes | yes | yes | 3.1s | 5,041 | $0.0021 |
| pol-05 | search_policies | yes | yes | yes | 3.4s | 4,870 | $0.0021 |
| pol-06 | search_policies | yes | yes | yes | 3.9s | 4,983 | $0.0021 |
| reg-02 | search_policies, search_policies | yes | yes | yes | 9.6s | 7,350 | $0.0031 |
| mt-01 | promo_impact, search_policies | yes | yes | yes | 4.4s | 5,178 | $0.0023 |
| mt-02 | search_policies, recommend_order | yes | yes | yes | 6.5s | 6,065 | $0.0027 |
| reg-01 | none | yes | yes | - | 2.2s | 2,114 | $0.0009 |
| ref-01 | search_policies | yes | yes | - | 5.2s | 5,131 | $0.0022 |
| ref-02 | none | yes | yes | - | 2.0s | 2,107 | $0.0009 |
| ref-03 | none | yes | yes | - | 1.6s | 2,103 | $0.0009 |
| v-01 | query_sales, promo_impact | yes | yes | - | 11.6s | 7,563 | $0.0033 |
| v-02 | promo_impact | yes | yes | - | 8.3s | 4,445 | $0.0019 |
| v-03 | promo_impact, search_policies | yes | yes | yes | 9.0s | 5,168 | $0.0023 |
| v-04 | search_policies | yes | yes | - | 5.6s | 5,159 | $0.0022 |
| v-05 | none | yes | yes | - | 2.1s | 2,107 | $0.0009 |
| v-06 | forecast_demand, query_sales | yes | yes | - | 8.1s | 5,590 | $0.0026 |

## History of this eval

Each step below changed the agent or the eval after seeing results, so the numbers are not directly comparable.

| Run | Questions | Passed | What happened |
|---|---|---|---|
| 1 | 30 | 25 (83%) | First run. 3 real agent failures: uplift worked out with SQL instead of `promo_impact` (`pr-03`); a different promotion analysed than the one the user named (`mt-01`); the retailer margin cap given as "our profit margin" (`ref-01`). 2 judge failures: extra detail penalised (`rec-03`) and non-policy claims flagged as ungrounded (`mt-02`). |
| 2 | 36 | 31 (86%) | Agent prompt tightened (always use `promo_impact` for uplift, use ids as given, no related-but-different figures, answer every part). Groundedness judge now must quote evidence, verified in code. 6 fresh variants added. Remaining failures were mostly the judge failing multi-part answers and two over-strict "no % anywhere" regexes on the margin questions. |
| 3 | 36 | 35 (97%) | Correctness judge redesigned as a checklist with quotes verified in code. The two margin regexes were removed (the agent now correctly says the retailer margin cap is not our margin). Rate-limit retries added after Azure returned HTTP 429 mid-run. The one failure (`ref-03`) was a bug in the eval: a conditional key fact ("any number is labelled an estimate") can never be present when no number is given. |
| 4 | 36 | see Summary | Final run, after fixing the `ref-03` key fact. |

Honest reading: questions were fixed after they failed, so the original 30 are partly tuned on; the 6 fresh variants were written before run 2 and not tuned on. With 36 questions, a difference of one question is within noise.

## Method and limitations

- Expected answers for data questions are computed when the eval runs (`truth_sql`, or a direct call of the tool), so they cannot go stale if the data is regenerated. Tool-based questions check that the agent reports the tool's numbers faithfully; the tools themselves are unit-tested separately.
- The judge is an LLM from the same family as the agent, so it can be lenient; regex checks back it up where the key fact is a simple phrase.
- One run per question at temperature 0. LLM output can still vary slightly between runs, and 30 questions is a small sample: treat a 1-2 question change as noise.
- Number matching reads any number in the answer; a coincidental match is possible for small expected values.
- Not measured yet: whether every number in an answer appears in a tool result (a general "no invented numbers" check).
