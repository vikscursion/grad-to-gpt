import math


class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._inputs = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(data=self.data + other.data, _children=(self, other), _op="+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(data=self.data * other.data, _children=(self, other), _op="*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        ordered_nodes = []
        visited = set()

        def add_after_inputs(node):
            if node not in visited:
                visited.add(node)
                for input_node in node._inputs:
                    add_after_inputs(input_node)
                ordered_nodes.append(node)

        add_after_inputs(self)
        self.grad = 1.0
        for node in reversed(ordered_nodes):
            node._backward()

    def tanh(self):
        out = Value(data=math.tanh(self.data), _children=(self,), _op="tanh")

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):  # handles 2 + x  ->  x + 2
        return self + other

    def __rmul__(self, other):  # handles 2 * x  ->  x * 2
        return self * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)
