# Learning Log — Grad-to-GPT

Entry format (from the program): date · hours · did/learned · what broke ·
retrieval/teach-back · next.

---

## 2026-07-02 — Day 1

- **Hours:** ~1
- **Did:** Wrote the project spec (`SPEC.md`) — defined the three components, each
  one's "done" criterion vs. a published reference, and the mastery bar. Stood up
  the repo skeleton (uv + ruff + pytest + GitHub Actions CI, green from commit 1).
  Learned what each component actually is: **autograd** = the gradient engine that
  makes learning possible (backprop); **MLP/MNIST** = the raw training loop with
  nothing hidden; **char-GPT** = a miniature transformer, the architecture behind
  ChatGPT.
- **Broke:** —
- **Teach-back (todo):** in one paragraph, explain what a gradient is and why
  training a network needs it.
- **Next:** Phase 3 — start the `Value` class in `src/grad_to_gpt/autograd/`.
  Watch Karpathy's micrograd video up to the point where you could attempt
  `backward()` yourself, then attempt it *before* looking at his code.
