"""Placeholders for the tests you'll write as you build each component.

Each is skipped until you implement it. When you reach that slice, delete the skip
and fill in the body. Skipped tests still show green in CI while documenting intent.
"""

import pytest


@pytest.mark.skip(reason="Component 1 (autograd): implement in phase 3")
def test_autograd_matches_torch():
    # Same inputs through your Value graph and through torch.autograd should
    # produce gradients that agree to ~1e-6.
    ...


@pytest.mark.skip(reason="Component 2 (MNIST): implement in phase 3")
def test_mnist_reaches_target_accuracy():
    # A short training run should exceed ~97% test accuracy.
    ...


@pytest.mark.skip(reason="Component 3, slice 1 (tokenizer): implement in phase 4")
def test_tokenizer_roundtrip():
    # decode(encode(s)) == s for sample strings.
    ...


@pytest.mark.skip(reason="Component 3, slice 2 (attention): implement in phase 4")
def test_causal_mask_blocks_future_tokens():
    # Changing a future token must not change the output at an earlier position.
    ...
