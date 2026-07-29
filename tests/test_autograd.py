"""Correctness tests for the from-scratch autograd engine (grad_to_gpt.autograd.engine).

Two independent checks of the gradients:
  1. Analytic grads from Value.backward() vs central finite differences
     (pure-Python ground truth, always runs in CI).
  2. Analytic grads vs torch.autograd (runs only when PyTorch is installed).
The test expression exercises +, *, tanh, a scalar, and reuse of `a`
(so gradient accumulation via += is covered too).
"""

import math

import pytest

from grad_to_gpt.autograd.engine import Value


def test_forward_values():
    a, b = Value(2.0), Value(3.0)
    assert (a + b).data == 5.0
    assert (a * b).data == 6.0
    assert math.isclose(Value(0.5).tanh().data, math.tanh(0.5))
    assert (2 * a).data == 4.0  # scalar on the left  -> __rmul__
    assert (a + 1).data == 3.0  # scalar on the right -> wrapped in Value


def _expr_value(a, b, c):
    return (a * b + a * c).tanh() + b * 2


def _expr_float(a, b, c):
    return math.tanh(a * b + a * c) + b * 2


def test_gradcheck_finite_differences():
    a, b, c = Value(1.5), Value(-2.0), Value(0.5)
    _expr_value(a, b, c).backward()
    analytic = {"a": a.grad, "b": b.grad, "c": c.grad}

    h = 1e-6
    base = (1.5, -2.0, 0.5)
    for i, k in enumerate("abc"):
        up, dn = list(base), list(base)
        up[i] += h
        dn[i] -= h
        numeric = (_expr_float(*up) - _expr_float(*dn)) / (2 * h)
        assert abs(analytic[k] - numeric) < 1e-6, (k, analytic[k], numeric)


def test_matches_pytorch():
    torch = pytest.importorskip("torch")  # skipped cleanly if torch isn't installed
    a, b, c = Value(1.5), Value(-2.0), Value(0.5)
    _expr_value(a, b, c).backward()

    ta = torch.tensor(1.5, requires_grad=True)
    tb = torch.tensor(-2.0, requires_grad=True)
    tc = torch.tensor(0.5, requires_grad=True)
    ((ta * tb + ta * tc).tanh() + tb * 2).backward()

    assert abs(a.grad - ta.grad.item()) < 1e-6
    assert abs(b.grad - tb.grad.item()) < 1e-6
    assert abs(c.grad - tc.grad.item()) < 1e-6
