# Grad-to-GPT

[![CI](https://github.com/<your-gh-username>/grad-to-gpt/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-gh-username>/grad-to-gpt/actions/workflows/ci.yml)

Building the deep-learning stack from scratch to understand it end to end:
a scalar **autograd engine** → an **MLP on MNIST** in raw PyTorch → a ~1M-param
**character-level GPT** trained on Tiny Shakespeare. Each component is verified
against a published reference.

> Full problem statement, "done" criteria, and mastery bar: **[SPEC.md](SPEC.md)**.

## Status

| Component | What it is | Reference | Done |
|---|---|---|:--:|
| 1. Autograd engine | scalar reverse-mode autodiff (micrograd-style) | Karpathy micrograd | ☐ |
| 2. MNIST MLP | hand-written training loop, raw PyTorch | ~97–98% baseline | ☐ |
| 3. Char-level GPT | tokenizer + attention + transformer, ~1M params | Karpathy nanoGPT | ☐ |

## Quickstart

```bash
# Install uv once:  https://docs.astral.sh/uv/

# Set up the environment (numpy + dev tools):
uv sync

# When you reach component 2/3, add the deep-learning deps:
uv sync --extra dl

# Run checks:
uv run ruff check .
uv run pytest -q
```

## Layout

```
src/grad_to_gpt/
  autograd/        component 1  (you build the Value class + backward pass)
  mnist/           component 2  (you build the MLP + hand-written loop)
  gpt/             component 3  (you build tokenizer, attention, transformer)
tests/             smoke test (green now) + skipped roadmap tests
SPEC.md            the one-page spec
LEARNING_LOG.md    daily / weekly log
```

## Results

_Filled in as each component ships — max gradient error vs PyTorch, MNIST test
accuracy, and char-GPT train/val loss vs the reference numbers._

## Notes

Learning project — ML/RL track, Stage 1, built from scratch on purpose. Each
component's write-up separates "what I built" from "what I borrowed."
