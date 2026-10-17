# Jagadish-dspy-prompt-optimizer

Programmatic prompt optimization in the DSPy spirit: self-improving
prompts without hand-tuning.

> Inspired by [DSPy](https://github.com/stanfordnlp/dspy) (Stanford NLP).
> This is an independent implementation built for study — it is not
> affiliated with, endorsed by, or a port of the DSPy project.

## The problem

Hand-tuned prompts don't survive contact with reality. The model gets
updated, the task drifts, an edge case appears — and someone re-opens
the prompt file and guesses again. Prompt engineering becomes folklore:
a collection of incantations nobody can explain or reproduce.

The DSPy insight: **stop tuning the prompt, optimize the program.**
Parametrize what can change (instructions, few-shot demonstrations),
define a metric, and let an optimizer search the space against real
data. Improvements compound instead of rotting.

## Architecture

```
dspyopt/
├── lm.py          LM protocol, FakeLM (deterministic test double),
│                  RealLMAdapter (live-client swap point)
├── program.py     Predict: signature + prompt template + LM + parser.
│                  Everything an optimizer may change lives here.
├── metrics.py     exact_match, contains_all, llm_judge_agreement,
│                  evaluate(program, dataset)
├── teleprompt.py  BootstrapFewShot (learn demos from correct traces),
│                  InstructionProposer (deterministic instruction mutations),
│                  optimize() (bootstrap → instruction search)
└── dataset.py     Synthetic sentiment + ticket-extraction tasks with
                   train/validation splits (all data fictional)
```

The loop is always **propose → evaluate → select**. See
[docs/how-it-works.md](docs/how-it-works.md).

## Quickstart

```python
from dspyopt import *

task = sentiment_task()
oracle = build_oracle(task["train"] + task["val"])

program = Predict(
    instruction="Classify the sentiment. Respond with exactly one of: positive, negative.",
    lm=FakeLM(oracle=oracle, seed=0),
    parse_fn=parse_label,
)

optimized, trace = optimize(program, task["train"], task["val"])
for stage in trace:
    print(stage["stage"], "->", stage["score"])
```

## Example optimization trace

On the synthetic sentiment task (8 validation reviews: easy, mixed-signal,
and sarcastic cases), `optimize()` produces:

| Stage | Val score | What happened |
|-------|-----------|---------------|
| baseline | 0.625 | Format-correct instruction; easy cases right, ambiguous ones wrong |
| bootstrap | 0.875 | Mined 1 correct training trace as a demo; mixed-signal cases fixed |
| instruction | 1.000 | Counterexample-hint mutation ("watch out for mixed signals and sarcasm") fixed the sarcastic case |

Final instruction (found by search, not by hand):

> Classify the sentiment. Respond with exactly one of: positive, negative.
> Watch out for mixed signals and sarcasm; the overall judgment determines
> the answer.

The trace is monotonic by construction — each stage keeps the best
program seen so far. Every decision is reproducible: same seed, same data,
same result.

## The FakeLM test double

All optimization runs offline against `FakeLM`, a deterministic seeded LM
that simulates three realistic behaviors (documented in `dspyopt/lm.py`):

1. **Instruction following** — clean, parseable output only when the
   instruction names the required format (or demos demonstrate it).
2. **Convention learning** — ambiguous inputs ("but…", "however…") are
   answered wrong zero-shot; few-shot demos teach the convention.
3. **Hint sensitivity** — sarcastic inputs need demos *plus* an
   instruction that names the pitfall.

## Production path

1. **Swap the LM.** Implement `RealLMAdapter.complete()` against your
   provider's chat API (OpenAI, Anthropic, Azure OpenAI — stub and
   example in `dspyopt/lm.py`), then pass it to `Predict` instead of
   `FakeLM`. Nothing else changes.
2. **Bring real data.** Replace the synthetic tasks with labeled
   train/validation sets from your domain. Keep the splits disjoint.
3. **Pick real metrics.** `exact_match` works for classification and
   structured extraction; wire `llm_judge_agreement` to a real judge
   LM for open-ended tasks.
4. **Scale the search.** The greedy demo selection and mutation
   operators here are the minimal viable versions — the extension
   points are `BootstrapFewShot._correct_traces` and
   `InstructionProposer.mutations`.

## Tests

```bash
python -m pytest tests/ -q
```

31 tests, all deterministic, no network, no API keys.

## CI

`.github/workflows/ci.yml` is prepared for the repo — add it via the
GitHub web UI (the API cannot push workflow files).

## Knowledge

Distilled notes and architecture diagrams: [knowledge/](knowledge/README.md)
