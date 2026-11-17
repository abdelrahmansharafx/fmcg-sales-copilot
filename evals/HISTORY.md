Each step below changed the agent or the eval after seeing results, so the numbers are not directly comparable.

| Run | Questions | Passed | What happened |
|---|---|---|---|
| 1 | 30 | 25 (83%) | First run. 3 real agent failures: uplift worked out with SQL instead of `promo_impact` (`pr-03`); a different promotion analysed than the one the user named (`mt-01`); the retailer margin cap given as "our profit margin" (`ref-01`). 2 judge failures: extra detail penalised (`rec-03`) and non-policy claims flagged as ungrounded (`mt-02`). |
| 2 | 36 | 31 (86%) | Agent prompt tightened (always use `promo_impact` for uplift, use ids as given, no related-but-different figures, answer every part). Groundedness judge now must quote evidence, verified in code. 6 fresh variants added. Remaining failures were mostly the judge failing multi-part answers and two over-strict "no % anywhere" regexes on the margin questions. |
| 3 | 36 | 35 (97%) | Correctness judge redesigned as a checklist with quotes verified in code. The two margin regexes were removed (the agent now correctly says the retailer margin cap is not our margin). Rate-limit retries added after Azure returned HTTP 429 mid-run. The one failure (`ref-03`) was a bug in the eval: a conditional key fact ("any number is labelled an estimate") can never be present when no number is given. |
| 4 | 36 | see Summary | Final run, after fixing the `ref-03` key fact. |

Honest reading: questions were fixed after they failed, so the original 30 are partly tuned on; the 6 fresh variants were written before run 2 and not tuned on. With 36 questions, a difference of one question is within noise.
