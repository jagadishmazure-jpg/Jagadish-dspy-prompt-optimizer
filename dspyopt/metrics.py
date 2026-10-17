"""Metrics: how we score a program on a dataset."""

from __future__ import annotations

import re
import string
from typing import Callable, Dict, List

from .program import Predict

Metric = Callable[[str, str], bool]


def exact_match(predicted: str, expected: str) -> bool:
    return predicted.strip() == expected.strip()


def contains_all(predicted: str, required: List[str]) -> bool:
    return all(fragment in predicted for fragment in required)


def _normalize(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return re.sub(r"\s+", " ", text).strip()


def llm_judge_agreement(predicted: str, expected: str) -> bool:
    """Deterministic stand-in for an LLM judge.

    Compares normalized forms (case, punctuation, whitespace ignored).
    In production this is where a real judge LM call would go; the
    heuristic keeps the test suite offline and deterministic.
    """
    return _normalize(predicted) == _normalize(expected)


def evaluate(
    program: Predict, dataset: List[Dict[str, str]], metric: Metric = exact_match
) -> Dict:
    """Score a program on a dataset.

    Returns {"score": float in [0, 1], "results": [per-example dicts]}.
    """
    results = []
    for example in dataset:
        out = program.forward({"input": example["input"]})
        correct = metric(out["output"], example["output"])
        results.append(
            {
                "input": example["input"],
                "predicted": out["output"],
                "expected": example["output"],
                "correct": correct,
            }
        )
    score = sum(r["correct"] for r in results) / len(results) if results else 0.0
    return {"score": score, "results": results}
