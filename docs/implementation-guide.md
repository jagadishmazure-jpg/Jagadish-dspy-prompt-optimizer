# Implementation guide: from offline build to Azure

> Status: **planned**. Nothing here has been deployed. The offline build in this
> repository is what exists today; this guide is how it would be taken to a
> real Azure environment.

## Azure service mapping

| Offline component | Azure service (planned) | Notes |
|---|---|---|
| `RealLMAdapter` | Azure OpenAI chat completions with Entra ID auth | Cache responses per prompt |
| Datasets | Labelled examples in Blob or an Azure AI Foundry dataset |  |
| Optimisation runs | Azure ML job or Container Apps job | Log each trace |
| Chosen prompt | Versioned in the repo or a prompt registry | Promote by PR |

## Steps

1. Implement `RealLMAdapter.complete()` against Azure OpenAI.
2. Bring a labelled dataset of at least a few hundred cases with disjoint splits.
3. Run `optimize()` as a batch job; store the trace and the winning instruction and demos.
4. Promote prompts through code review and re-run on model upgrades.

## Guardrails to keep when going live

- Train and validation splits are disjoint; demos come only from training data.
- Stage selection never accepts a lower score than the previous stage.
- Same seed and data give the same result.
- The live LM adapter is a stub that raises until it is wired up, so nothing calls a model by accident.

## What to measure before calling it production-ready

- Validation and held-out test scores per stage on real data.
- Cost (tokens) per optimisation run.
- Score stability across model versions.
