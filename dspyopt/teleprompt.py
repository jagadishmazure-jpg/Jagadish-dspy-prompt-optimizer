"""Teleprompters: optimizers that improve programs against data.

The loop is always propose -> evaluate -> select:

- `BootstrapFewShot` mines correct zero-shot traces from the training
  set as candidate demos, then greedily keeps the ones that raise the
  validation score.
- `InstructionProposer` mutates a seed instruction with deterministic
  operators and keeps the best-scoring variant on validation.
- `optimize` runs bootstrap, then instruction search, keeping the best
  program at every stage (trace scores are monotonically non-decreasing).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .metrics import Metric, evaluate, exact_match
from .program import Predict


# -- instruction mutation operators (deterministic) ------------------------
def add_format_rule(instruction: str) -> str:
    return instruction + " Output ONLY the answer in the required format. No explanation, no extra text."


def add_constraint(instruction: str) -> str:
    return instruction + " Be decisive and consider the overall meaning of the whole text."


def add_counterexample_hint(instruction: str) -> str:
    return (
        instruction
        + " Watch out for mixed signals and sarcasm; the overall judgment determines the answer."
    )


def add_deliberation_hint(instruction: str) -> str:
    return instruction + " Think step by step about what the author truly means."


DEFAULT_MUTATIONS = [
    add_format_rule,
    add_constraint,
    add_counterexample_hint,
    add_deliberation_hint,
]


class BootstrapFewShot:
    """Learn few-shot demos from correct training traces.

    Runs the program on the training set, keeps traces whose output
    matches the gold label as candidate demos, then greedily adds
    candidates that strictly improve the validation score (up to
    `max_demos`). Deterministic: ties keep the earlier candidate.
    """

    def __init__(self, max_demos: int = 4, metric: Metric = exact_match):
        self.max_demos = max_demos
        self.metric = metric

    def _correct_traces(
        self, program: Predict, trainset: List[Dict[str, str]]
    ) -> List[Tuple[str, str]]:
        candidates = []
        for example in trainset:
            out = program.forward({"input": example["input"]})
            if self.metric(out["output"], example["output"]):
                candidates.append((example["input"], example["output"]))
        return candidates

    def compile(
        self, program: Predict, trainset: List[Dict[str, str]], valset: List[Dict[str, str]]
    ) -> Tuple[Predict, List[Dict]]:
        candidates = self._correct_traces(program, trainset)
        current_demos = list(program.demos)
        best_program = program.with_demos(current_demos)
        best_score = evaluate(best_program, valset, self.metric)["score"]
        trace = [
            {
                "stage": "bootstrap",
                "step": 0,
                "n_demos": len(current_demos),
                "val_score": best_score,
                "action": f"collected {len(candidates)} correct traces from train",
            }
        ]
        step = 0
        for cand_in, cand_out in candidates:
            if len(current_demos) >= self.max_demos:
                break
            step += 1
            trial = program.with_demos(current_demos + [(cand_in, cand_out)])
            score = evaluate(trial, valset, self.metric)["score"]
            if score > best_score:
                current_demos.append((cand_in, cand_out))
                best_program = trial
                best_score = score
                action = "kept demo (val improved)"
            else:
                action = "skipped demo (no val gain)"
            trace.append(
                {
                    "stage": "bootstrap",
                    "step": step,
                    "n_demos": len(current_demos),
                    "val_score": round(score, 4),
                    "action": action,
                }
            )
        return best_program, trace


class InstructionProposer:
    """Search instruction variants via deterministic mutations.

    Scores the seed plus each mutation on the validation set and keeps
    the best (ties keep the earlier candidate, so the seed wins ties).
    """

    def __init__(self, mutations=None, metric: Metric = exact_match):
        self.mutations = list(mutations) if mutations else list(DEFAULT_MUTATIONS)
        self.metric = metric

    def candidates(self, seed_instruction: str) -> List[str]:
        seen = []
        for cand in [seed_instruction] + [m(seed_instruction) for m in self.mutations]:
            if cand not in seen:
                seen.append(cand)
        return seen

    def compile(
        self,
        program: Predict,
        trainset: List[Dict[str, str]],
        valset: List[Dict[str, str]],
        seed_instruction: str = None,
    ) -> Tuple[Predict, List[Dict]]:
        del trainset  # instruction search is validated, not trained
        seed = seed_instruction if seed_instruction is not None else program.instruction
        trace = []
        best_instruction, best_score = seed, -1.0
        for i, cand in enumerate(self.candidates(seed)):
            score = evaluate(program.with_instruction(cand), valset, self.metric)["score"]
            trace.append(
                {
                    "stage": "instruction",
                    "step": i,
                    "val_score": round(score, 4),
                    "action": "kept" if score > best_score else "rejected",
                    "instruction": cand,
                }
            )
            if score > best_score:
                best_instruction, best_score = cand, score
        return program.with_instruction(best_instruction), trace


def optimize(
    program: Predict,
    trainset: List[Dict[str, str]],
    valset: List[Dict[str, str]],
    max_demos: int = 4,
    metric: Metric = exact_match,
) -> Tuple[Predict, List[Dict]]:
    """Run bootstrap, then instruction search; keep the best at each stage.

    Returns (optimized_program, trace). Trace stage scores are
    monotonically non-decreasing by construction.
    """
    trace: List[Dict] = []
    baseline = evaluate(program, valset, metric)["score"]
    trace.append({"stage": "baseline", "score": round(baseline, 4)})

    prog_b, btrace = BootstrapFewShot(max_demos=max_demos, metric=metric).compile(
        program, trainset, valset
    )
    score_b = evaluate(prog_b, valset, metric)["score"]
    if score_b < baseline:
        prog_b, score_b = program, baseline
    trace.append({"stage": "bootstrap", "score": round(score_b, 4), "detail": btrace})

    proposer = InstructionProposer(metric=metric)
    prog_i, itrace = proposer.compile(prog_b, trainset, valset)
    score_i = evaluate(prog_i, valset, metric)["score"]
    if score_i < score_b:
        prog_i, score_i = prog_b, score_b
    trace.append({"stage": "instruction", "score": round(score_i, 4), "detail": itrace})

    return prog_i, trace
