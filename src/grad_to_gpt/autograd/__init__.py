"""Component 1: scalar reverse-mode autograd engine (micrograd-style).

`Value` builds a computation graph as arithmetic is performed; `backward()`
then applies the chain rule via a topological traversal of that graph.
Supported ops: ``+``, ``*``, ``tanh``, and scalar operands on either side.

Gradients are verified in ``tests/test_autograd.py`` against central finite
differences (agree to ~1e-10) and, when ``torch`` is installed, against
``torch.autograd``.

Reference: Karpathy micrograd.
"""

from .engine import Value

__all__ = ["Value"]
