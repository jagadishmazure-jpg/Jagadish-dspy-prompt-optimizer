"""Parametrized programs: signature + prompt template + LM + parser.

A `Predict` is the unit that optimizers improve. Everything the
optimizer may change — instruction, demos — lives on the program, so
`teleprompt` never touches LM internals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

from .lm import LM

DEFAULT_TEMPLATE = "{instruction}\n\n{demos}Input: {input}\nOutput:"

VALID_LABELS = ("positive", "negative")


def parse_label(raw: str) -> str:
    """Strict label parse: only a bare label counts."""
    cleaned = raw.strip().lower().rstrip(".")
    return cleaned if cleaned in VALID_LABELS else "unparseable"


def parse_json(raw: str) -> str:
    """Strict JSON parse: the output must contain a JSON object."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return "unparseable"
    try:
        obj = json.loads(raw[start : end + 1])
    except json.JSONDecodeError:
        return "unparseable"
    if not isinstance(obj, dict) or not obj:
        return "unparseable"
    return json.dumps(obj, sort_keys=True)


def render_demos(demos: List[Tuple[str, str]]) -> str:
    parts = []
    for i, (demo_in, demo_out) in enumerate(demos, start=1):
        parts.append(f"Example {i}:\nInput: {demo_in}\nOutput: {demo_out}\n")
    return "\n".join(parts) + ("\n" if parts else "")


@dataclass
class Predict:
    """A parametrized prediction program.

    Args:
        instruction: natural-language task instruction (optimizer-mutable).
        lm: any `LM` — FakeLM in tests, RealLMAdapter in production.
        parse_fn: maps raw LM text -> scored output string.
        template: prompt template with {instruction}, {demos}, {input}.
        demos: few-shot demonstrations as (input, output) pairs.
        signature: declared input/output fields (documentation + routing).
    """

    instruction: str
    lm: LM
    parse_fn: Callable[[str], str] = parse_label
    template: str = DEFAULT_TEMPLATE
    demos: List[Tuple[str, str]] = field(default_factory=list)
    signature: Dict[str, List[str]] = field(
        default_factory=lambda: {"inputs": ["input"], "outputs": ["output"]}
    )

    def render_prompt(self, inputs: Dict[str, str]) -> str:
        return self.template.format(
            instruction=self.instruction,
            demos=render_demos(self.demos),
            input=inputs["input"],
        )

    def forward(self, inputs: Dict[str, str]) -> Dict[str, str]:
        prompt = self.render_prompt(inputs)
        raw = self.lm.complete(prompt)
        return {"output": self.parse_fn(raw), "raw": raw, "prompt": prompt}

    def with_demos(self, demos: List[Tuple[str, str]]) -> "Predict":
        return Predict(
            instruction=self.instruction,
            lm=self.lm,
            parse_fn=self.parse_fn,
            template=self.template,
            demos=list(demos),
            signature=self.signature,
        )

    def with_instruction(self, instruction: str) -> "Predict":
        return Predict(
            instruction=instruction,
            lm=self.lm,
            parse_fn=self.parse_fn,
            template=self.template,
            demos=list(self.demos),
            signature=self.signature,
        )
