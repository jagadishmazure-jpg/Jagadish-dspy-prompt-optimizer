"""Tests for dspyopt — all deterministic, no network, no keys."""

import json

import pytest

from dspyopt import (
    BootstrapFewShot,
    FakeLM,
    InstructionProposer,
    Predict,
    RealLMAdapter,
    build_oracle,
    contains_all,
    evaluate,
    exact_match,
    extract_task,
    llm_judge_agreement,
    optimize,
    parse_json,
    parse_label,
    sentiment_task,
)


# ---------------------------------------------------------------- helpers
SENT_SEED = "Classify the sentiment. Respond with exactly one of: positive, negative."
VAGUE_SEED = "Read the ticket and answer."


def sentiment_program(instruction=SENT_SEED, seed=0, demos=None):
    task = sentiment_task()
    oracle = build_oracle(task["train"] + task["val"])
    lm = FakeLM(oracle=oracle, seed=seed)
    return Predict(instruction=instruction, lm=lm, parse_fn=parse_label, demos=demos or [])


def extract_program(instruction=VAGUE_SEED, seed=0, demos=None):
    task = extract_task()
    oracle = build_oracle(task["train"] + task["val"])
    lm = FakeLM(oracle=oracle, seed=seed)
    return Predict(instruction=instruction, lm=lm, parse_fn=parse_json, demos=demos or [])


# ---------------------------------------------------------------- FakeLM
def test_fakelm_deterministic_same_seed():
    task = sentiment_task()
    oracle = build_oracle(task["train"] + task["val"])
    a, b = FakeLM(oracle=oracle, seed=7), FakeLM(oracle=oracle, seed=7)
    prompt = "Classify.\n\nInput: I love this.\nOutput:"
    assert a.complete(prompt) == b.complete(prompt)


def test_fakelm_deterministic_across_calls():
    lm = sentiment_program().lm
    prompt = "Classify.\n\nInput: I love this.\nOutput:"
    assert lm.complete(prompt) == lm.complete(prompt)


def test_fakelm_easy_correct_zero_shot():
    prog = sentiment_program()
    out = prog.forward({"input": "I absolutely love this blender."})
    # "I absolutely love this blender." is not in the oracle -> unknown
    assert out["output"] == "unparseable"
    out = prog.forward({"input": "Great value for the price, highly recommend to anyone."})
    assert out["output"] == "positive"


def test_fakelm_hard_wrong_zero_shot():
    prog = sentiment_program()
    out = prog.forward({"input": "The design is beautiful but the battery life is disappointing."})
    assert out["output"] == "positive"  # flipped: gold is negative


def test_fakelm_hard_correct_with_demos():
    prog = sentiment_program(
        demos=[("The design is beautiful but the battery life is disappointing.", "negative")]
    )
    out = prog.forward({"input": "Nice look and feel, but it stopped working after a month."})
    assert out["output"] == "negative"


def test_fakelm_tricky_needs_hint():
    tricky = "Sarcastic applause for the slowest delivery I have ever seen."
    prog = sentiment_program(demos=[("Great value for the price, highly recommend to anyone.", "positive")])
    assert prog.forward({"input": tricky})["output"] == "positive"  # wrong without hint
    hinted = sentiment_program(
        instruction=SENT_SEED + " Watch out for mixed signals and sarcasm; the overall judgment determines the answer.",
        demos=[("Great value for the price, highly recommend to anyone.", "positive")],
    )
    assert hinted.forward({"input": tricky})["output"] == "negative"


def test_fakelm_vague_instruction_bad_format():
    prog = sentiment_program(instruction="Tell me about the sentiment.")
    out = prog.forward({"input": "Great value for the price, highly recommend to anyone."})
    assert out["raw"].startswith("The sentiment of this review is")
    assert out["output"] == "unparseable"


def test_fakelm_json_format_instruction():
    prog = extract_program(instruction="Read the ticket. Respond in JSON.")
    out = prog.forward({"input": "How do I reset my password for the dashboard?"})
    obj = json.loads(out["output"])
    assert obj == {"category": "account", "priority": "low"}


def test_fakelm_json_vague_instruction_prose():
    prog = extract_program(instruction="Read the ticket and answer.")
    out = prog.forward({"input": "How do I reset my password for the dashboard?"})
    assert "{" not in out["raw"]
    assert out["output"] == "unparseable"


# ---------------------------------------------------------------- Predict
def test_predict_renders_prompt_with_demos():
    prog = sentiment_program(demos=[("I love it.", "positive")])
    prompt = prog.render_prompt({"input": "It broke."})
    assert "Example 1:" in prompt
    assert SENT_SEED in prompt
    assert "Input: It broke.\nOutput:" in prompt


def test_predict_parses_output():
    prog = sentiment_program()
    out = prog.forward({"input": "Great value for the price, highly recommend to anyone."})
    assert out["output"] == "positive"
    assert "prompt" in out and "raw" in out


def test_predict_signature_fields():
    prog = sentiment_program()
    assert prog.signature == {"inputs": ["input"], "outputs": ["output"]}


def test_predict_with_demos_and_instruction_are_immutable_copies():
    prog = sentiment_program()
    p2 = prog.with_demos([("a", "b")])
    p3 = prog.with_instruction("new")
    assert prog.demos == [] and prog.instruction == SENT_SEED
    assert p2.demos == [("a", "b")] and p3.instruction == "new"


# ---------------------------------------------------------------- metrics
def test_exact_match():
    assert exact_match("positive", "positive")
    assert exact_match("  positive ", "positive")
    assert not exact_match("positive", "negative")


def test_contains_all():
    assert contains_all("priority high, category billing", ["high", "billing"])
    assert not contains_all("priority high", ["high", "billing"])


def test_llm_judge_agreement():
    assert llm_judge_agreement("Positive!", "positive")
    assert llm_judge_agreement("  POSITIVE. ", "positive")
    assert not llm_judge_agreement("positive", "negative")


def test_evaluate_scoring_and_shape():
    prog = sentiment_program()
    task = sentiment_task()
    result = evaluate(prog, task["val"])
    assert 0.0 <= result["score"] <= 1.0
    assert len(result["results"]) == len(task["val"])
    first = result["results"][0]
    assert set(first) == {"input", "predicted", "expected", "correct"}
    assert result["score"] == pytest.approx(5 / 8)


# ---------------------------------------------------------------- BootstrapFewShot
def test_bootstrap_improves_over_zeroshot():
    prog = sentiment_program()
    task = sentiment_task()
    before = evaluate(prog, task["val"])["score"]
    opt, trace = BootstrapFewShot(max_demos=4).compile(prog, task["train"], task["val"])
    after = evaluate(opt, task["val"])["score"]
    assert after > before
    assert after == pytest.approx(7 / 8)


def test_bootstrap_respects_max_demos():
    prog = sentiment_program()
    task = sentiment_task()
    opt, _ = BootstrapFewShot(max_demos=2).compile(prog, task["train"], task["val"])
    assert len(opt.demos) <= 2


def test_bootstrap_demos_were_correct_traces():
    prog = sentiment_program()
    task = sentiment_task()
    oracle = build_oracle(task["train"] + task["val"])
    opt, _ = BootstrapFewShot(max_demos=4).compile(prog, task["train"], task["val"])
    assert opt.demos, "expected at least one demo"
    for demo_in, demo_out in opt.demos:
        assert oracle[demo_in] == demo_out


def test_bootstrap_trace_records_steps():
    prog = sentiment_program()
    task = sentiment_task()
    _, trace = BootstrapFewShot(max_demos=4).compile(prog, task["train"], task["val"])
    assert trace[0]["stage"] == "bootstrap"
    assert all("val_score" in step and "action" in step for step in trace)


# ---------------------------------------------------------------- InstructionProposer
def test_instruction_mutations_valid():
    proposer = InstructionProposer()
    cands = proposer.candidates(SENT_SEED)
    assert cands[0] == SENT_SEED
    assert len(cands) == len(set(cands))  # distinct
    assert all(isinstance(c, str) and c for c in cands)
    assert len(cands) == 5  # seed + 4 mutations


def test_instruction_proposer_improves_format():
    prog = extract_program(instruction=VAGUE_SEED)  # 0.0: nothing parses
    task = extract_task()
    before = evaluate(prog, task["val"])["score"]
    assert before == 0.0
    opt, trace = InstructionProposer().compile(prog, task["train"], task["val"])
    after = evaluate(opt, task["val"])["score"]
    assert after > before
    assert "required format" in opt.instruction or "JSON" in opt.instruction


def test_instruction_best_wins_validation():
    prog = extract_program(instruction=VAGUE_SEED)
    task = extract_task()
    proposer = InstructionProposer()
    opt, trace = proposer.compile(prog, task["train"], task["val"])
    best_trace_score = max(step["val_score"] for step in trace)
    assert round(evaluate(opt, task["val"])["score"], 4) == best_trace_score


# ---------------------------------------------------------------- optimize
def test_optimize_end_to_end():
    prog = sentiment_program()
    task = sentiment_task()
    baseline = evaluate(prog, task["val"])["score"]
    opt, trace = optimize(prog, task["train"], task["val"])
    final = evaluate(opt, task["val"])["score"]
    assert final >= baseline
    assert final == pytest.approx(1.0)
    assert baseline == pytest.approx(5 / 8)


def test_optimize_trace_monotonic():
    prog = sentiment_program()
    task = sentiment_task()
    _, trace = optimize(prog, task["train"], task["val"])
    stages = [t["stage"] for t in trace]
    assert stages == ["baseline", "bootstrap", "instruction"]
    scores = [t["score"] for t in trace]
    assert all(b >= a for a, b in zip(scores, scores[1:])), scores


def test_optimize_returns_program_and_trace():
    prog = sentiment_program()
    task = sentiment_task()
    opt, trace = optimize(prog, task["train"], task["val"])
    assert isinstance(opt, Predict)
    assert isinstance(trace, list) and len(trace) == 3
    assert trace[2]["detail"], "instruction stage should record candidate trials"


# ---------------------------------------------------------------- dataset
def test_dataset_splits_disjoint():
    for task_fn in (sentiment_task, extract_task):
        task = task_fn()
        train_inputs = {ex["input"] for ex in task["train"]}
        val_inputs = {ex["input"] for ex in task["val"]}
        assert train_inputs.isdisjoint(val_inputs)


def test_dataset_splits_nonempty():
    for task_fn in (sentiment_task, extract_task):
        task = task_fn()
        assert len(task["train"]) >= 8 and len(task["val"]) >= 6


def test_dataset_labels_valid():
    task = sentiment_task()
    for ex in task["train"] + task["val"]:
        assert ex["output"] in ("positive", "negative")
    task = extract_task()
    for ex in task["train"] + task["val"]:
        obj = json.loads(ex["output"])
        assert set(obj) == {"priority", "category"}


# ---------------------------------------------------------------- adapter stub
def test_real_lm_adapter_stub():
    adapter = RealLMAdapter()
    with pytest.raises(NotImplementedError, match="live LM client"):
        adapter.complete("hello")
