"""dspyopt — programmatic prompt optimization in the DSPy spirit.

Self-improving prompts without hand-tuning: parametrize the program,
then let optimizers search instructions and few-shot demonstrations
against a validation score.

Inspired by DSPy (Stanford NLP); this is an independent implementation
for study, not affiliated with the DSPy project.
"""

from .dataset import build_oracle, extract_task, sentiment_task
from .lm import LM, FakeLM, RealLMAdapter
from .metrics import contains_all, evaluate, exact_match, llm_judge_agreement
from .program import DEFAULT_TEMPLATE, Predict, parse_json, parse_label
from .teleprompt import BootstrapFewShot, InstructionProposer, optimize

__all__ = [
    "LM",
    "FakeLM",
    "RealLMAdapter",
    "Predict",
    "parse_label",
    "parse_json",
    "DEFAULT_TEMPLATE",
    "exact_match",
    "contains_all",
    "llm_judge_agreement",
    "evaluate",
    "BootstrapFewShot",
    "InstructionProposer",
    "optimize",
    "sentiment_task",
    "extract_task",
    "build_oracle",
]
