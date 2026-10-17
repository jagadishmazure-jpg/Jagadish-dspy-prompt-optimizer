"""Language-model interface and test doubles.

`FakeLM` is a deterministic, seeded stand-in for a real LM. It simulates
three realistic LM behaviors so the optimization machinery can be tested
offline:

1. **Instruction following (format).** It emits a clean, parseable answer
   only when the instruction names the required format (or when few-shot
   demos demonstrate it). Vague instructions yield chatty prose that the
   parser rejects.
2. **Convention learning (few-shot).** "Hard" inputs (mixed signals such
   as "but"/"however") are answered wrong zero-shot; with at least one
   demo in the prompt it applies the demonstrated convention.
3. **Hint sensitivity (edge cases).** "Tricky" inputs (sarcasm) need demos
   *plus* an instruction that names the pitfall ("mixed signals",
   "sarcasm", "overall judgment").

`RealLMAdapter` is the swap-in point for a live client.
"""

from __future__ import annotations

import json
import random
import re
from typing import Dict, List, Optional, Protocol


class LM(Protocol):
    """Minimal LM interface: prompt in, text out."""

    def complete(self, prompt: str) -> str: ...


# Markers that make a synthetic input "hard" (ambiguous) or "tricky".
HARD_MARKERS = ["but", "however", "although", "not sure", "maybe", "might be"]
TRICKY_MARKERS = ["sarcastic", "yeah right", "oh great"]

# Instruction keywords FakeLM reacts to (documented simulation behavior).
BARE_FORMAT_KEYWORDS = ["only the answer", "exactly one of", "no explanation"]
JSON_FORMAT_KEYWORDS = ["json", "required format"]
HINT_KEYWORDS = ["mixed signals", "sarcasm", "overall judgment"]


class FakeLM:
    """Deterministic seeded LM test double.

    Args:
        oracle: maps input text -> correct output text. The double "knows"
            the task but behaves imperfectly, per the rules above.
        seed: fixes all stochastic choices; a fresh RNG per call keeps
            output independent of call order.
    """

    def __init__(self, oracle: Optional[Dict[str, str]] = None, seed: int = 0):
        self.oracle = dict(oracle or {})
        self.seed = seed

    # -- LM interface ----------------------------------------------------
    def complete(self, prompt: str) -> str:
        rng = random.Random(self.seed)
        instruction, demo_count, query = self._parse_prompt(prompt)
        gold = self.oracle.get(query)
        if gold is None:
            return "I don't know."

        kind = "json" if gold.strip().startswith("{") else "label"
        lowered = query.lower()
        tricky = any(m in lowered for m in TRICKY_MARKERS)
        hard = tricky or any(m in lowered for m in HARD_MARKERS)

        content_ok = self._content_ok(tricky, hard, demo_count, instruction)
        answer = gold if content_ok else self._corrupt(gold, kind, rng)

        if self._format_ok(kind, instruction, demo_count):
            return answer
        return self._deformat(answer, kind)

    # -- prompt parsing --------------------------------------------------
    @staticmethod
    def _parse_prompt(prompt: str):
        demo_count = len(re.findall(r"Example \d+:", prompt))
        matches = list(re.finditer(r"Input:\s*(.*?)\nOutput:", prompt, re.DOTALL))
        query = matches[-1].group(1).strip() if matches else ""
        instruction = re.split(r"Example 1:|Input:", prompt, maxsplit=1)[0].strip()
        return instruction, demo_count, query

    # -- simulated behaviors ---------------------------------------------
    @staticmethod
    def _format_ok(kind: str, instruction: str, demo_count: int) -> bool:
        if demo_count > 0:
            return True  # demos demonstrate the output format
        text = instruction.lower()
        if kind == "json":
            return any(k in text for k in JSON_FORMAT_KEYWORDS)
        return any(k in text for k in BARE_FORMAT_KEYWORDS)

    @staticmethod
    def _content_ok(tricky: bool, hard: bool, demo_count: int, instruction: str) -> bool:
        if not hard:
            return True
        if demo_count == 0:
            return False
        if tricky:
            text = instruction.lower()
            return any(k in text for k in HINT_KEYWORDS)
        return True

    @staticmethod
    def _corrupt(gold: str, kind: str, rng: random.Random) -> str:
        if kind == "json":
            try:
                obj = json.loads(gold)
            except json.JSONDecodeError:
                return gold
            if "priority" in obj:
                obj["priority"] = "low" if obj["priority"] == "high" else "high"
            elif obj:
                obj.pop(rng.choice(sorted(obj.keys())))
            return json.dumps(obj)
        low = gold.strip().lower()
        if low == "positive":
            return "negative"
        if low == "negative":
            return "positive"
        choices = [c for c in ("positive", "negative") if c != low]
        return rng.choice(choices or [gold])

    @staticmethod
    def _deformat(answer: str, kind: str) -> str:
        if kind == "json":
            try:
                obj = json.loads(answer)
                p = obj.get("priority", "?")
                c = obj.get("category", "?")
                return f"Looking at this ticket, I would say the priority is {p} and it belongs in {c}."
            except (json.JSONDecodeError, AttributeError):
                return f"My take on this ticket: {answer}."
        return f"The sentiment of this review is {answer}."

    def demo_outputs(self) -> List[str]:
        """All oracle outputs (handy for building demo pools in tests)."""
        return list(self.oracle.values())


class RealLMAdapter:
    """Swap-in point for a live LM client (OpenAI, Anthropic, etc.).

    Not used in tests — calling it requires a configured client, an API
    key, and network access. To go live, subclass or wrap this and
    implement `complete()` against your provider's chat API, e.g.::

        class OpenAIAdapter(RealLMAdapter):
            def complete(self, prompt: str) -> str:
                resp = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content
    """

    def __init__(self, client=None):
        self.client = client

    def complete(self, prompt: str) -> str:
        raise NotImplementedError(
            "RealLMAdapter.complete requires a configured live LM client "
            "(and API key). Pass a client to the constructor and implement "
            "complete() against your provider's chat API. See the docstring."
        )
