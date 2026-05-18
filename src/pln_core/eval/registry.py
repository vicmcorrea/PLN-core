"""Tiny registry helper used by the evaluation factories.

Keeps the dataset and analyzer factories independent and easy to extend
without forcing callers to import every implementation module by hand.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """Name based registry for factory callables."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._items:
                raise ValueError(f"{self._kind} '{name}' already registered")
            self._items[name] = factory
            return factory

        return decorator

    def create(self, name: str, **kwargs: object) -> T:
        try:
            factory = self._items[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._items)) or "<none>"
            raise KeyError(
                f"unknown {self._kind} '{name}'. Available: {available}"
            ) from exc
        return factory(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)
