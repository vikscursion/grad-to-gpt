"""Grad-to-GPT: build the deep-learning stack from scratch.

Three components, built in order (see SPEC.md):
    1. autograd - a scalar reverse-mode autodiff engine (micrograd-style)
    2. mnist    - an MLP trained on MNIST with a hand-written PyTorch loop
    3. gpt      - a ~1M-param character-level GPT (transformer) on Tiny Shakespeare
"""

__version__ = "0.1.0"
