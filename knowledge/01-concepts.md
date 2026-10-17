# DSPy prompt optimization: concepts

## The thesis: prompts are programs — compile them, don't hand-craft them

Hand-written prompts are untested code. DSPy's move is to treat the
prompt as a *parameter* and optimize it programmatically against a
validation metric: bootstrap few-shot demonstrations from training
traces, propose instruction variants, keep what strictly improves the
score. This repo implements that machinery end-to-end.

## 1. FakeLM (`dspyopt/lm.py`) — the test double that makes it real

A deterministic fake language model with three documented behaviors
(instruction-following, convention learning, hint sensitivity). Because
the LM is a seam (`dspyopt/program.py` programs against the LM
interface), the entire optimization machinery is real and testable
offline — swap in a real LM and the optimizers don't change.

Interview line: *"The LM is a dependency-injected interface. Everything
that matters — the search, the selection, the monotonicity — is verified
against the fake, so the machinery is proven before it ever touches an
API."*

## 2. BootstrapFewShot (`dspyopt/teleprompt.py`)

Mines correct training traces as demonstration candidates, then keeps
only what *strictly improves* validation score. The selection rule is
the point: demos are data, and data is selected by the metric, not by
vibes.

## 3. InstructionProposer (`dspyopt/teleprompt.py`)

Four deterministic mutation operators over the instruction text;
best-by-validation selection. Prompt engineering becomes a search
problem with a score function.

## 4. The real trace (from the README)

```
0.625 → 0.875 → 1.000
```

across optimization stages — monotonically non-decreasing *by
construction*, because nothing is kept unless it strictly improves
validation. That's the property you'd defend in an interview: the
optimizer can't make things worse.

## Module map

| Module | Role |
|---|---|
| `dspyopt/lm.py` | FakeLM test double (3 documented behaviors) |
| `dspyopt/program.py` | programs against the LM interface |
| `dspyopt/teleprompt.py` | BootstrapFewShot + InstructionProposer |
| `dspyopt/metrics.py` | validation scoring |
| `dspyopt/dataset.py` | train/validation splits |

## Production path

Replace FakeLM with a real LM client behind the same interface; keep the
teleprompters unchanged. Add a real metric (exact-match → LLM judge),
and the 0.625 → 1.000 trace becomes a real optimization run.
