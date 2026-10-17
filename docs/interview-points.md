# Interview points

Short, honest talking points for Jagadish-dspy-prompt-optimizer. Every number quoted is from the
real run shown in the README.

- Optimise the program, not the prompt: parameters, a metric and a search.
- Bootstrap fixed the mixed-signal cases; the counterexample hint fixed sarcasm.
- I am clear the 1.0 is on 8 cases against a deterministic double built to respond to those fixes.
- The trace is monotonic because each stage keeps the best program so far.

## Likely follow-up questions

- How do you avoid overfitting the validation set? (A separate held-out test set and more data.)
- When would you fine-tune instead? (When prompts plateau and you have enough labelled data.)
- What does DSPy add beyond this? (LLM-proposed instructions, more optimisers, multi-module programs.)
