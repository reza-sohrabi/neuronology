# neuronology

Can raw MLP neurons find and remove a planted backdoor? The spec is `PLAN.md`.
Per-milestone summaries are in `results/M<n>/SUMMARY.md`.

## Setup (Mac or RunPod)

```
uv sync                       # Python 3.11, deps pinned in uv.lock (public PyPI index)
uv run pre-commit install     # ruff lint + format on commit
uv run pytest                 # M0 hook tests; downloads the pilot model on first run
```

Useful environment variables:

- `NEURONOLOGY_MODEL`: the model the tests use (default `Qwen/Qwen3.5-0.8B`).
- `NEURONOLOGY_DEVICE`: force `cpu`, `mps` or `cuda` (default: cuda > mps > cpu).

## Layout (current)

- `src/neuronology/model.py`: loading, device and dtype handling, architecture info, chat templating with thinking off.
- `src/neuronology/hooks.py`: capture of neuron activations (input of `mlp.down_proj`), `z_gate`/`z_up`, and `intervene(fn)`.
- `tests/`: the four Phase 0 tests.
- `runpod/`: job bootstrap, watchdog/auto-stop wrapper, results sync. These are files only; nothing has been run on RunPod.
- `budget.json`: the spend ledger (caps from PLAN.md section 3).

`secrets/`, `artifacts/` and `.env` are gitignored and must never be committed.
