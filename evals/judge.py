"""LLM-as-judge for free-text answers: correctness against key facts, and RAG groundedness.

Design: the LLM never makes the final call. It fills in a checklist (key fact -> present?, with
an exact quote as evidence), and our code verifies every quote really appears in the text and
decides the verdict. Limitations: the judge is an LLM from the same family as the agent, so it
can still misread; regex checks back it up wherever a key fact is a simple phrase.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field

import openai

from evals.retry import with_retries
from src import llm

CORRECTNESS_SYSTEM = """You grade a sales assistant's answer against a checklist.
For EACH key fact, decide whether the ANSWER states it. Different wording, units or number \
formatting are fine if the meaning is the same. Give "evidence": an EXACT short quote copied \
from the ANSWER that states the fact ("" if it does not).
Also list "contradictions": statements in the ANSWER that DIRECTLY conflict with the REFERENCE. \
Extra information (more numbers, examples, caveats, other parts of a multi-part question) is NOT \
a contradiction. Use only the reference, not your own knowledge.
Reply with one JSON object: {"facts": [{"fact": "...", "present": true or false, \
"evidence": "..."}], "contradictions": ["..."]}"""

GROUNDEDNESS_SYSTEM = """You check whether an assistant's answer is grounded in retrieved \
policy passages.
Find the claims in the ANSWER that are about COMPANY POLICY: rules, terms, thresholds, durations, \
approvers, price points stated in the policy documents. For each one give:
  "claim": the claim in a few words,
  "supported": true only if the PASSAGES state it (a table row counts),
  "evidence": an EXACT short quote copied from the PASSAGES that supports it ("" if unsupported).
Do NOT list claims about sales figures, forecasts, order quantities, outlets or promotion results: \
those come from other tools, not from the passages. Ignore greetings and formatting.
Reply with one JSON object: {"claims": [{"claim": "...", "supported": true or false, \
"evidence": "..."}]}"""


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9.%]+", text.lower())


def quote_in_passages(quote: str, passages: str) -> bool:
    """Is the judge's evidence quote really in the passages? (ignores markdown and spacing)"""
    clean = lambda s: " ".join(re.sub(r"[|*_`#]", " ", s.lower()).split())  # noqa: E731
    if clean(quote) and clean(quote) in clean(passages):
        return True
    quote_tokens, passage_tokens = _tokens(quote), set(_tokens(passages))
    found = sum(token in passage_tokens for token in quote_tokens)
    return bool(quote_tokens) and found / len(quote_tokens) >= 0.9


@dataclass
class Verdict:
    passed: bool | None  # None = the judge could not run (API unavailable): skipped
    detail: str


@dataclass
class Judge:
    """Calls the LLM and keeps a running total of the tokens the judging itself used."""

    complete: Callable = llm.complete
    usage: llm.Usage = field(default_factory=llm.Usage)

    def _ask(self, system: str, user: str) -> dict:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        with llm.track_usage() as used:
            reply = with_retries(lambda: self.complete(messages, json_mode=True))
        self.usage.calls += used.calls
        self.usage.prompt_tokens += used.prompt_tokens
        self.usage.completion_tokens += used.completion_tokens
        return parse_json(reply)

    def correctness(
        self, question: str, answer: str, reference: str, key_facts: list[str]
    ) -> Verdict:
        """Correct = every key fact is present with a quote found in the answer, no contradiction.

        The model fills in the checklist; this code makes the decision. That keeps the verdict
        auditable and stops the judge from failing a good answer for harmless extra detail.
        """
        facts = key_facts or [reference]
        checklist = "\n".join(f"- {fact}" for fact in facts)
        user = (
            f"QUESTION: {question}\n\nREFERENCE: {reference}\n\n"
            f"KEY FACTS:\n{checklist}\n\nANSWER:\n{answer}"
        )
        try:
            result = self._ask(CORRECTNESS_SYSTEM, user)
            problems = []
            for item in result["facts"]:
                if not item["present"]:
                    problems.append(f"missing: {item['fact']}")
                elif not quote_in_passages(item.get("evidence", ""), answer):
                    problems.append(f"evidence not found in answer: {item['fact']}")
            if len(result["facts"]) < len(facts):
                problems.append("judge skipped some key facts")
            problems += [f"contradiction: {c}" for c in result.get("contradictions") or []]
        except openai.APIError as error:  # rate limit / outage: not the agent's fault
            return Verdict(None, f"judge unavailable: {type(error).__name__}")
        except (ValueError, KeyError, TypeError) as error:
            return Verdict(False, f"judge error: {error}")
        if problems:
            return Verdict(False, "; ".join(problems))
        return Verdict(True, f"all {len(facts)} key facts present, no contradictions")

    def groundedness(self, question: str, answer: str, passages: str) -> Verdict:
        """Grounded = every policy claim is marked supported AND its quote is in the passages."""
        user = f"QUESTION: {question}\n\nPASSAGES:\n{passages}\n\nANSWER:\n{answer}"
        try:
            claims = self._ask(GROUNDEDNESS_SYSTEM, user)["claims"]
            problems = []
            for claim in claims:
                if not claim["supported"]:
                    problems.append(f"unsupported: {claim['claim']}")
                elif not quote_in_passages(claim.get("evidence", ""), passages):
                    problems.append(f"evidence not found in passages: {claim['claim']}")
        except openai.APIError as error:
            return Verdict(None, f"judge unavailable: {type(error).__name__}")
        except (ValueError, KeyError, TypeError) as error:
            return Verdict(False, f"judge error: {error}")
        if problems:
            return Verdict(False, "; ".join(problems))
        return Verdict(True, f"{len(claims)} policy claims, all supported by quoted evidence")


def parse_json(reply: str) -> dict:
    """Parse the judge's JSON reply, tolerating text around it."""
    match = re.search(r"\{.*\}", reply, flags=re.DOTALL)
    if not match:
        raise ValueError("no JSON object in judge reply")
    return json.loads(match.group(0))
