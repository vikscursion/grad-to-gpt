# Grad-to-GPT — Project Spec (Stage 1, Project 1)

**Owner:** Viksit Singh · **Started:** 2026-07-02 · **Target ship:** 2026-07-16 (2 weeks)
**Compute:** Google Colab (free/Pro GPU) · **Corpus:** Tiny Shakespeare · **Status:** Phase 2 (skeleton)

---

## What it is

Build the deep-learning stack **from scratch, bottom to top**, in order to *understand* it — not to lean on a framework's magic. Three components, each a rung on the ladder, each verified against a published reference. The point is not novelty; it's that every interview answer I give later comes from having built these myself.

## Components & "done" criteria

**1. Scalar autograd engine** — `src/grad_to_gpt/autograd/`
A micrograd-style `Value` class: builds a computation graph as you do arithmetic, then a topological-sort backward pass applies the chain rule through `+`, `*`, `tanh`/`ReLU`, etc. Use it to train a tiny MLP on a toy dataset.
- **Done:** my gradients match PyTorch's `autograd` on identical inputs (agree to ~1e-6); the toy MLP drives its loss to near-zero on a separable problem.
- **Reference:** Karpathy *micrograd*.

**2. MLP on MNIST in raw PyTorch** — `src/grad_to_gpt/mnist/`
A hand-written training loop — forward → loss → `zero_grad` → backward → `step` — with **no** `nn.Sequential` / high-level `Trainer` shortcuts the first time. The loop stays visible so I can see exactly what PyTorch is doing for me.
- **Done:** ≥ ~97% MNIST test accuracy; loss curve shape matches a standard MLP baseline.
- **Reference:** standard PyTorch MNIST MLP (~97–98%).

**3. Char-level GPT, ~1M params** — `src/grad_to_gpt/gpt/`
Char tokenizer (then a from-scratch BPE toy), single- then multi-head **causal self-attention**, a transformer block (LayerNorm + residuals + MLP), stacked into a small GPT, trained on Tiny Shakespeare, with temperature/top-k sampling.
- **Done:** validation loss within ~10% of Karpathy's nanoGPT/makemore numbers on Tiny Shakespeare; generates coherent Shakespeare-like text.
- **Reference:** Karpathy *nanoGPT* / "Let's build GPT."

## What I measure

For each component I log final numbers and put them in the README results table: (1) max gradient error vs PyTorch; (2) MNIST test accuracy + loss curve; (3) char-GPT train/val loss curves vs the reference numbers.

## Mastery bar — how I'll know I actually learned it

- Explain backprop through a matmul on a whiteboard.
- Rebuild single-head attention from a blank page in < 20 minutes.
- Char-GPT loss within ~10% of reference across a couple of runs.

## Explicitly out of scope (this project)

Multi-GPU / distributed training; a production BPE beyond a from-scratch toy; model serving/deployment (that is the Stage 3–4 flagship); any corpus other than Tiny Shakespeare.

## Build order — the 7-phase arc

1. **Specify** — this document.
2. **Skeleton** — repo, uv env, pytest, ruff, CI green, README stub. ← *today*
3. **Core from scratch** — autograd `Value` + backward pass; then the MNIST loop by hand.
4. **Vertical slices** — tokenizer → attention (single → multi → causal, with a masking test) → full transformer block → GPT on Shakespeare → sampling. Blank-page rebuild of each slice's core afterward.
5. **Harden** — tokenizer round-trip test, masking test, shape-contract tests; loss logging (CSV or W&B); profile one bottleneck.
6. **Ship** — README with architecture diagram, "built vs. borrowed," loss curves, sample generations, and "5 things that silently break transformers" (from my own bugs).
7. **Extract** — write-up "Attention from first principles"; mock interview with Claude on backprop-through-attention.

## Risks & notes (I'm new to PyTorch)

- Tensors / autograd / training loops get taught just-in-time right before the step that needs them.
- **20-minute rule:** stuck > 20 min → switch Claude into debugging-coach mode, do **not** tutorial-hop.
- Colab sessions time out — checkpoint the char-GPT and keep the dataset/model small so a lost session costs minutes, not hours.

---

*This spec is the phase-1 deliverable. It is intentionally short: if I can't say what "done" means before writing code, I don't understand the problem yet.*
