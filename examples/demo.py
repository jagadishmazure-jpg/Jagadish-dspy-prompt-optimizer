"""Optimise a sentiment classifier and a ticket extractor against the deterministic FakeLM."""
from dspyopt import FakeLM, Predict, build_oracle, optimize, parse_json, parse_label
from dspyopt.dataset import extract_task, sentiment_task


def run(name, task, instruction, parse_fn):
    oracle = build_oracle(task["train"] + task["val"])
    program = Predict(instruction=instruction, lm=FakeLM(oracle=oracle, seed=0), parse_fn=parse_fn)
    optimized, trace = optimize(program, task["train"], task["val"])
    print(f"--- {name}: {len(task['train'])} train / {len(task['val'])} validation examples")
    for stage in trace:
        print(f"  {stage['stage']:<12} val score {stage['score']}")
    print("  demos kept:", len(optimized.demos))
    print("  final instruction:", optimized.instruction)


run("sentiment", sentiment_task(),
    "Classify the sentiment. Respond with exactly one of: positive, negative.", parse_label)
run("ticket extraction", extract_task(),
    "Extract the ticket priority and category as JSON.", parse_json)
