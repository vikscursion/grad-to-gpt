"""Component 1: scalar autograd engine (micrograd-style).

YOU build this in phase 3. Target shape (fill in the bodies yourself):

    class Value:
        '''A single scalar, its gradient, and the op that produced it.'''
        def __init__(self, data, _children=(), _op=""): ...
        def __add__(self, other): ...
        def __mul__(self, other): ...
        def tanh(self): ...
        def backward(self):
            # 1. build a topological ordering of the graph
            # 2. set self.grad = 1.0
            # 3. walk nodes in reverse, calling each node's local _backward()

Done (see SPEC.md): gradients match torch.autograd to ~1e-6 on identical inputs,
and a tiny MLP built on Value drives a toy loss to ~0.

Reference: Karpathy micrograd. Don't read its source until you've attempted the
backward pass yourself — the struggle is the learning.
"""
