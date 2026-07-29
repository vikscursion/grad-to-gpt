# Grad-to-GPT

[![CI](https://github.com/vikscursion/grad-to-gpt/actions/workflows/ci.yml/badge.svg)](https://github.com/vikscursion/grad-to-gpt/actions/workflows/ci.yml)

Building the deep-learning stack from scratch to understand it end to end:
a scalar **autograd engine** → an **MLP on MNIST** in raw PyTorch → a ~1M-param
**character-level GPT** trained on Tiny Shakespeare. Each component is verified
against a published reference.

> Full problem statement, "done" criteria, and mastery bar: **[SPEC.md](SPEC.md)**.

## Status

| Component | What it is | Reference | Done |
|---|---|---|:--:|
| 1. Autograd engine | scalar reverse-mode autodiff (micrograd-style) | Karpathy micrograd | ✓ |
| 2. MNIST MLP | hand-written training loop, raw PyTorch | ~97–98% baseline | ☐ |
| 3. Char-level GPT | tokenizer + attention + transformer, ~1M params | Karpathy nanoGPT | ☐ |

✓ **Component 1 — autograd engine:** the `Value` class (`+`, `*`, `-`, `tanh`, scalar
operands) and the topological-sort `backward()` pass are built and gradient-verified,
and a small MLP built on it (`nn.py`) trains a toy dataset to near-zero loss
(`train_toy.py`). See Results.

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
  autograd/        component 1  — engine.py (Value + backward ✓), nn.py (Neuron/Layer/MLP), train_toy.py (demo)
  mnist/           component 2  (you build the MLP + hand-written loop)
  gpt/             component 3  (you build tokenizer, attention, transformer)
tests/             autograd gradient-check + smoke (green) + skipped roadmap tests
SPEC.md            the one-page spec
LEARNING_LOG.md    daily / weekly log
```

## Results

**Component 1 — autograd engine (verified 2026-07-27).** Analytic gradients from
`Value.backward()` match central finite differences to **1.7e-10** on an expression
combining `+`, `*`, `tanh`, a scalar, and a reused input (so gradient accumulation
via `+=` is covered). A PyTorch-parity check
(`tests/test_autograd.py::test_matches_pytorch`) runs once `torch` is installed.

A 41-parameter MLP built on the engine (`nn.py`) trains a 4-example toy set from
loss **7.46 → 0.009** in 100 steps (`train_toy.py`), with predictions matching the
targets — so the engine trains a real network end to end.

_MNIST test accuracy and char-GPT train/val loss land here as components 2–3 ship._

## Notes

Learning project — ML/RL track, Stage 1, built from scratch on purpose. Each
component's write-up separates "what I built" from "what I borrowed."
```
