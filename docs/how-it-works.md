# How it works: the optimization loop

Every optimizer in `dspyopt/teleprompt.py` runs the same loop:

```
propose  →  evaluate  →  select
```

## 1. Propose

Generate candidate changes to the program. Two kinds:

- **Demonstrations** (`BootstrapFewShot`): run the current program on
  the training set and keep the traces it got *right* as candidate
  few-shot demos. The insight: the program's own correct behavior,
  captured as examples, teaches it the task's conventions — including
  conventions the instruction never stated.
- **Instructions** (`InstructionProposer`): mutate the seed instruction
  with deterministic operators — add a format rule, add a constraint,
  add a counterexample hint, add a deliberation hint. No LLM needed to
  propose; the operators are fixed and inspectable.

## 2. Evaluate

Score each candidate on the **validation set** — never the training set.
`evaluate()` runs the program end-to-end (render prompt → call LM →
parse output) and applies the metric. The score is the only thing the
optimizer trusts.

## 3. Select

Keep the candidate that strictly improves the validation score; ties
keep the earlier candidate (deterministic). `optimize()` chains the two
optimizers and keeps the best program at every stage, so trace scores
are monotonically non-decreasing by construction.

## Why this beats manual tuning

| Manual tuning | Programmatic optimization |
|---|---|
| One person, one intuition | Search over candidates, scored on data |
| "It worked on my three examples" | Validated on a held-out set |
| Improvements rot when the model changes | Re-run the optimizer on the new model |
| Folklore ("always say 'think step by step'") | Mutations kept only if they score |

## Worked example: the sentiment trace

Seed instruction: *"Classify the sentiment. Respond with exactly one of:
positive, negative."* — format-correct, but silent on ambiguity.

1. **Baseline (0.625).** Easy reviews correct. Mixed-signal reviews
   ("The design is beautiful but…") get the wrong label: without a
   demonstrated convention, the LM follows surface words.
2. **Bootstrap (0.875).** One correct training trace becomes a demo.
   The demo teaches the convention — overall judgment wins over mixed
   signals — and fixes the ambiguous validation cases. The sarcastic
   review still fails: no demo covered sarcasm.
3. **Instruction (1.000).** The counterexample-hint mutation adds
   *"Watch out for mixed signals and sarcasm; the overall judgment
   determines the answer."* That sentence, found by search, fixes the
   last case.

Note what happened: the final prompt contains knowledge no human wrote
down — the sarcasm pitfall was discovered because a mutation covered it
and the validation score moved. That is the whole game.
