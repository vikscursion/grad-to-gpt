"""Trivial test so CI is green from day one.

Real tests arrive with each build slice — see tests/test_roadmap.py for the list.
"""

import grad_to_gpt


def test_package_imports():
    assert grad_to_gpt.__version__
