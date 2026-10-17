"""dspyopt — programmatic prompt optimization in the DSPy spirit.

Self-improving prompts without hand-tuning: parametrize the program,
then let optimizers search instructions and few-shot demonstrations
against a validation score.

Inspired by DSPy (Stanford NLP); this is an independent implementation
for study, not affiliated with the DSPy project.
"""

from .lm import LM, FakeLM, RealLMAdapter
from .program import Predict, parse_label, parse_json, DEFAULT_TEMPLATE
from .metrics import exact_match, contains_all, llm_judge_agreement, evaluate
from .teleprompt import BootstrapFewShot, InstructionProposer, optimize
from .dataset import sentiment_task, extract_task, build_oracle

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
