"""Component 3: a ~1M-param character-level GPT on Tiny Shakespeare.

YOU build this in phases 3-4, in slices (see SPEC.md):
    slice 1: char tokenizer (encode/decode; write the round-trip test)
    slice 2: attention - single head from the equation, then multi-head, then
             causal masking (write the test: a masked/future position must not
             change the output at an earlier position)
    slice 3: transformer block (LayerNorm + residual + MLP), stacked into a GPT
    slice 4: sampling (temperature, top-k)

Needs the deep-learning deps:  uv sync --extra dl
Done (see SPEC.md): val loss within ~10% of Karpathy's Tiny Shakespeare numbers;
generates coherent Shakespeare-like text.
"""
