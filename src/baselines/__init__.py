"""Baseline planning methods B0 and B1."""

from .b0 import B0
from .b1 import B1
from .method import Method

METHODS: dict[str, Method] = {method.name: method for method in (B0(), B1())}


def get_method(name: str) -> Method:
    if name not in METHODS:
        raise ValueError(f"unknown method {name!r}; known methods: {sorted(METHODS)}")
    return METHODS[name]
